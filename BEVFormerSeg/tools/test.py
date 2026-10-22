# Entry point for BEVFormerSeg evaluation.
#
# Differences vs tools/test_v1.py (the legacy HDMapNet-style entry point):
#   * Removed hard-coded `sys.path.insert(...)` for a foreign workspace.
#   * Imports single_gpu_test / multi_gpu_test from apis/test.py which now
#     computes 7-class IoU under BEVFusion class names.
#   * Diffusion-only dataset defaults are passed only to diffusion datasets.
#   * Removed the `assert False` that aborted `--out` saving in v1.
#
# Usage:
#   python tools/test.py \
#       projects/configs/bevformer/bevformer_tiny_seg_v2.py \
#       <checkpoint>.pth \
#       --eval bbox
#
# Distributed:
#   ./tools/dist_test.sh \
#       projects/configs/bevformer/bevformer_tiny_seg_v2.py \
#       <checkpoint>.pth 8 \
#       --launcher pytorch --eval bbox

import argparse
import importlib
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.apis.test import (multi_gpu_test,
                                               single_gpu_test)
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description='BEVFormerSeg V2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out',
                        help='output result file in pickle format (.pkl)')
    parser.add_argument('--fuse-conv-bn', action='store_true',
                        help='Fuse conv and bn for slight inference speedup')
    parser.add_argument('--format-only', action='store_true',
                        help='Format the output without evaluation')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics (e.g. bbox)')
    parser.add_argument('--show', action='store_true',
                        help='show seg results (saved per-sample image)')
    parser.add_argument('--show-dir',
                        help='directory where seg images will be saved')
    parser.add_argument('--gpu-collect', action='store_true',
                        help='collect results to GPU instead of CPU tmpdir')
    parser.add_argument('--tmpdir',
                        help='tmp directory used for collecting results from '
                             'multiple workers (CPU collect)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='set deterministic options for CUDNN')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override config (key=val)')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='custom options for dataset.evaluate()')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _import_plugin(cfg, config_path):
    """Trigger plugin module import so PIPELINES / models are registered."""
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
    else:
        plugin_dir = osp.dirname(config_path)
    module_parts = plugin_dir.rstrip('/').split('/')
    module_path = '.'.join(p for p in module_parts if p)
    print(f'[plugin] importing {module_path}')
    importlib.import_module(module_path)


def _build_dataset(cfg):
    """Pass diffusion defaults only to datasets that accept them."""
    default_args = {}
    dataset_type = (cfg.data.test.get('type', '')
                    if isinstance(cfg.data.test, dict) else '')
    if isinstance(dataset_type, str) and dataset_type.startswith(
            'CustomNuScenesDiffusionDataset'):
        for key in ('point_cloud_range', 'use_3d_bbox', 'num_classes', 'num_bboxes'):
            if cfg.get(key, None) is not None:
                default_args[key.replace('point_cloud_range', 'pc_range')] = cfg[key]
    return build_dataset(cfg.data.test, default_args=default_args or None)


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, (
            'Please specify at least one operation (save/eval/format/show '
            'the results) with --out / --eval / --format-only / --show / '
            '--show-dir')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    _import_plugin(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # samples_per_gpu — single sample per GPU during test
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
            ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test)
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

    dataset = _build_dataset(cfg)
    # num_workers=0: multi_gpu_test/single_gpu_test build the CUDA `iou_metric`
    # and run the CUDA model in the main process before iterating this loader,
    # so forked workers would inherit live CUDA tensors and crash with
    # "CUDA error: initialization error" when torn down. See the matching note
    # in projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py.
    test_workers_per_gpu = cfg.data.get('val_workers_per_gpu', 0)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=test_workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

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

    if not distributed:
        model = MMDataParallel(model.cuda(0), device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        # MMCV 1.x expects this flag, which newer PyTorch no longer creates.
        if not hasattr(model, '_use_replicated_tensor_module'):
            model._use_replicated_tensor_module = False
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)

    kwargs = {} if args.eval_options is None else args.eval_options
    kwargs['jsonfile_prefix'] = osp.join(
        'test',
        osp.basename(args.config).rsplit('.', 1)[0],
        time.ctime().replace(' ', '_').replace(':', '_'))

    if args.format_only:
        dataset.format_results(outputs, **kwargs)

    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ('interval', 'tmpdir', 'start', 'gpu_collect',
                    'save_best', 'rule'):
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
