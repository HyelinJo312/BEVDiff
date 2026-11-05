# Copyright (c) OpenMMLab. All rights reserved.
# Stage 2 (DiffBEVDet4D) training entry: BEVDet's tools/train.py plus the teacher setup of
# BEVFormer/tools/train_onlyseg.py (build_frozen_teacher, --load_from, --unet_checkpoint_dir).
# tools/train.py is left unchanged for the BEVDet4D baseline (C0).
#   - load_from: checkpoint of the frozen BEV target (model_target) ONLY; it is cleared after
#     loading, so the student starts from init_weights() (ImageNet backbone), as in the flagship.
#   - bev_diffuser_cfg.unet_cfg is taken from the top-level `unet` (Stage 1 architecture).
from __future__ import division
import argparse
import copy
import os
import sys
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.runner.checkpoint import _load_checkpoint

BEVDET_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if BEVDET_ROOT not in sys.path:
    sys.path.insert(0, BEVDET_ROOT)  # `projects` = BEVDet/projects

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

from projects.bevdet_plugin.apis import custom_train_model  # noqa: E402
from bevdiffuser_onlyseg import BEVDiffuser, resolve_unet_checkpoint  # noqa: E402  (tools/)

# Student-only modules of DiffBEVDet4D; a BEVDet4D target checkpoint has no such keys.
STUDENT_ONLY_PREFIXES = ('mgd_generation.', 'mgd_align.', 'bev_distill_proj.')


def build_frozen_teacher(cfg, model, logger):
    """Frozen BEV target (from load_from) and the frozen semantic-only teacher.

    As in BEVFormer/tools/train_onlyseg.py: model_target = deepcopy(model) loaded from
    cfg.load_from, after which cfg.load_from is cleared so the runner does not load it into
    the student. The target load is strict except for the student-only modules.
    """
    model_target = None
    bev_diffuser = None
    if cfg.get('bev_diffuser_cfg') is not None:
        if not cfg.get('load_from'):
            raise ValueError('Stage 2 needs load_from = checkpoint of the frozen BEV target (C0)')
        model_target = copy.deepcopy(model)
        checkpoint = _load_checkpoint(cfg.load_from, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        missing, unexpected = model_target.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith(STUDENT_ONLY_PREFIXES)]
        if bad_missing or unexpected:
            raise RuntimeError(f'model_target load from {cfg.load_from}: missing {bad_missing[:5]} '
                               f'(+{max(len(bad_missing) - 5, 0)}), unexpected {unexpected[:5]}')
        logger.info(f'model_target <- {cfg.load_from} (strict; student-only keys not in the '
                    f'checkpoint: {len(missing)}). The student is NOT loaded from it.')
        cfg.load_from = None
        model_target.requires_grad_(False)
        model_target.eval()
        if hasattr(model_target, 'with_prev'):
            model_target.with_prev = True  # the trained target always uses the previous frame
        bev_diffuser = BEVDiffuser(**cfg.bev_diffuser_cfg)
        bev_diffuser.requires_grad_(False)
        logger.info(f'bev_diffuser <- {cfg.bev_diffuser_cfg.unet_checkpoint_dir} '
                    f'(noise t={bev_diffuser.noise_timesteps}, denoise={bev_diffuser.denoise_timesteps}, '
                    f'DDIM steps={bev_diffuser.num_inference_steps}, cfg_scale={bev_diffuser.cfg_scale}, '
                    f'classifier guidance={bev_diffuser.use_classifier_guidence})')
    return model_target, bev_diffuser


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--unet_checkpoint_dir', help='Stage 1 checkpoint dir (or its unet/ dir / weight file)')
    parser.add_argument(
        '--load_from', help='checkpoint of the frozen BEV target (model_target only)')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if cfg.get('bev_diffuser_cfg') is not None:
        # Config merging may produce separate copies of the nested UNet config.
        # The top-level UNet, including CLI overrides, is the source of truth.
        cfg.bev_diffuser_cfg.unet_cfg = copy.deepcopy(cfg.unet)
        if args.unet_checkpoint_dir is not None:
            cfg.bev_diffuser_cfg.unet_checkpoint_dir = args.unet_checkpoint_dir
        resolve_unet_checkpoint(cfg.bev_diffuser_cfg.get('unet_checkpoint_dir'))
    if args.load_from is not None:
        if not osp.isfile(args.load_from):
            raise FileNotFoundError(f'load_from: {args.load_from}')
        cfg.load_from = args.load_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)

    # remove objectsample augmentation in the 2nd stage
    if cfg.get('two_stage', False):
        assert len(cfg.workflow) == 1
        assert cfg.data.train.dataset.pipeline[5].type =='ObjectSample'
        assert len(cfg.data.train.dataset.pipeline)==15
        cfg.data.train.dataset.pipeline = \
            [cfg.data.train.dataset.pipeline[kid] for kid in
             [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]]
        cfg.data.train.dataset.pipeline.append(
            dict(type='Collect3D',
                 keys=['points', 'gt_bboxes_3d', 'gt_depth',
                'gt_labels_3d', 'img_inputs']))
        datasets.append(build_dataset(cfg.data.train))
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    model_target, bev_diffuser = build_frozen_teacher(cfg, model, logger)

    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta,
        model_target=model_target,  # frozen BEVDet4D (clean BEV x0)
        bev_diffuser=bev_diffuser)  # frozen semantic-only teacher


if __name__ == '__main__':
    main()
