# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# Derived from tools/test.py (BEVFormer). Runs a single inference pass over the
# test set, then reports nuScenes detection metrics (mAP / NDS) broken down by
# scene condition: overall / sunny / rainy / day / night.
#
# Conditions are read from the nuScenes scene `description` field. 'sunny/rainy'
# and 'day/night' are two independent partitions of the val set.
#
# Usage mirrors test.py (launch distributed via dist_test_by_condition.sh):
#   torchrun --nproc_per_node=4 tools/test_by_condition.py CONFIG CHECKPOINT \
#       --launcher pytorch

import argparse
import os
import os.path as osp
import time
import warnings

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.datasets.nuscnes_eval import (
    NuScenesEval_custom,
    evaluate_by_scene_condition,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-scene-condition nuScenes evaluation (single inference pass).')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--fuse-conv-bn', action='store_true',
        help='fuse conv and bn for a small speedup')
    parser.add_argument('--gpu-collect', action='store_true',
                        help='whether to use gpu to collect results.')
    parser.add_argument('--tmpdir',
                        help='tmp dir for collecting results from workers')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='set deterministic options for CUDNN backend.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import plugin modules, registry will be updated.
    if hasattr(cfg, 'plugin') and cfg.plugin and hasattr(cfg, 'plugin_dir'):
        import importlib
        _module_path = '.'.join(os.path.dirname(cfg.plugin_dir).split('/'))
        importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader. Diffusion configs need extra default_args; vanilla
    # BEVFormer configs (CustomNuScenesDataset) don't accept them, so only pass
    # them when the config actually defines the diffusion-specific fields.
    if 'use_3d_bbox' in cfg:
        dataset_default_args = {
            'pc_range': cfg.point_cloud_range,
            'use_3d_bbox': cfg.use_3d_bbox,
            'num_classes': cfg.num_classes,
            'num_bboxes': cfg.num_bboxes,
        }
    else:
        dataset_default_args = None
    dataset = build_dataset(cfg.data.test, default_args=dataset_default_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    assert distributed, 'This script requires distributed launch (use the .sh wrapper).'
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    # single inference pass (predictions are condition-independent)
    outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    # format predictions to a nuScenes results json once.
    config_name = osp.splitext(osp.basename(args.config))[0]
    jsonfile_prefix = osp.join('test', config_name, 'by_condition',
                               time.ctime().replace(' ', '_').replace(':', '_'))
    result_files, _ = dataset.format_results(outputs, jsonfile_prefix=jsonfile_prefix)
    result_path = result_files['pts_bbox'] if isinstance(result_files, dict) \
        else result_files

    # build the evaluator once (loads gt + preds into all_gt / all_preds),
    # then evaluate per scene condition.
    from nuscenes import NuScenes
    nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=True)
    eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
    output_dir = osp.join(*osp.split(result_path)[:-1])
    nusc_eval = NuScenesEval_custom(
        nusc,
        config=dataset.eval_detection_configs,
        result_path=result_path,
        eval_set=eval_set_map[dataset.version],
        output_dir=output_dir,
        verbose=False,
        overlap_test=getattr(dataset, 'overlap_test', False),
        data_infos=dataset.data_infos,
    )
    evaluate_by_scene_condition(nusc_eval, nusc, result_path=result_path)


if __name__ == '__main__':
    main()
