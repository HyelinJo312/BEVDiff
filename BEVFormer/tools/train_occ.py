# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

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
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info, init_dist)
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet.core import EvalHook
from mmdet.datasets import replace_ImageToTensor
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version

sys.path.append(osp.dirname(osp.abspath(__file__)) + '/..')

from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook  # noqa: E402
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEVFormer Occ3D')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', '--work_dir', dest='work_dir')
    parser.add_argument('--load-from', '--load_from', dest='load_from')
    parser.add_argument('--resume-from', '--resume_from', dest='resume_from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int)
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true')
    parser.add_argument('--report_to', default=None)
    parser.add_argument('--tracker_project_name', default='Occ3D-BEVFormer')
    parser.add_argument('--tracker_run_name', default=None)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified.')
    if args.options:
        warnings.warn('--options is deprecated; use --cfg-options instead.')
        args.cfg_options = args.options
    return args


def import_custom_modules(cfg, config_path):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)

    if not cfg.get('plugin', False):
        return

    import importlib

    module_dir = cfg.get('plugin_dir', osp.dirname(config_path))
    module_parts = osp.dirname(module_dir).split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path += f'.{part}'
    print(module_path)
    importlib.import_module(module_path)


def get_logger_name(cfg):
    """Name under which this run's logger is registered.

    Both main() and train_occ_model() must use it: mmcv hands back a logger
    without any handler when asked for a name that merely shares a prefix with
    an already initialised one ('mmdet3d' vs 'mmdet'), which silently drops
    every log record written through it.
    """
    return 'mmseg' if cfg.model.type in ['EncoderDecoder3D'] else 'mmdet'


def add_logger_hook(cfg, hook_type, init_kwargs=None):
    hooks = cfg.log_config.setdefault('hooks', [])
    if any(hook.get('type') == hook_type for hook in hooks):
        return
    hook_cfg = dict(type=hook_type)
    if init_kwargs is not None:
        hook_cfg['init_kwargs'] = init_kwargs
    hooks.append(hook_cfg)


def configure_runtime(cfg, args):
    if args.tracker_run_name is None:
        args.tracker_run_name = osp.splitext(osp.basename(args.config))[0]

    if args.work_dir is not None:
        cfg.work_dir = osp.join(args.work_dir, args.tracker_run_name)
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.load_from is not None and osp.isfile(args.load_from):
        cfg.load_from = args.load_from

    cfg.gpu_ids = args.gpu_ids if args.gpu_ids is not None else range(
        1 if args.gpus is None else args.gpus)

    if (digit_version(TORCH_VERSION) == digit_version('1.8.1')
            and cfg.optimizer['type'] == 'AdamW'):
        cfg.optimizer['type'] = 'AdamW2'
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    if args.report_to == 'tensorboard':
        add_logger_hook(cfg, 'TensorboardLoggerHook')
    elif args.report_to == 'wandb':
        add_logger_hook(
            cfg,
            'WandbLoggerHook',
            init_kwargs=dict(
                project=args.tracker_project_name,
                name=args.tracker_run_name,
                id=args.tracker_run_name))


def build_train_datasets(cfg):
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    return datasets


def train_occ_model(model,
                    datasets,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    # Keyword args matter here: get_root_logger(cfg.log_level) would bind the
    # level to `log_file` and return an unconfigured logger, so every training
    # log line would vanish while only the .log.json kept being written.
    logger = get_root_logger(
        log_level=cfg.log_level, name=get_logger_name(cfg))
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler)
        for dataset in datasets
    ]

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    optimizer = build_optimizer(model, cfg.optimizer)
    if 'runner' not in cfg:
        cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.total_epochs)
        warnings.warn('Please set `runner` in the config.', UserWarning)
    elif 'total_epochs' in cfg:
        assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    runner.timestamp = timestamp

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None))

    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    if validate:
        register_eval_hook(runner, cfg, distributed)

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def register_eval_hook(runner, cfg, distributed):
    val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    if val_samples_per_gpu > 1:
        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)

    val_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    eval_cfg = cfg.get('evaluation', {})
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_cfg['jsonfile_prefix'] = osp.join(
        cfg.work_dir, 'val',
        time.ctime().replace(' ', '_').replace(':', '_'))
    eval_hook = CustomDistEvalHook if distributed else EvalHook
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg))


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_custom_modules(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    configure_runtime(cfg, args)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=get_logger_name(cfg))

    meta = dict()
    env_info = '\n'.join([f'{k}: {v}' for k, v in collect_env().items()])
    logger.info('Environment info:\n' + '-' * 60 + '\n' + env_info + '\n' +
                '-' * 60)
    logger.info(f'Distributed training: {distributed}')
    if distributed:
        logger.info(f'num of gpus: {world_size}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    logger.info(f'Model:\n{model}')

    datasets = build_train_datasets(cfg)
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE
            if hasattr(datasets[0], 'PALETTE') else None)
    model.CLASSES = datasets[0].CLASSES

    train_occ_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    main()
