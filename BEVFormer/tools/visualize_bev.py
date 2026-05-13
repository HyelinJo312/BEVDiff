# Copyright (c) 2025
# SPDX-License-Identifier: AGPL-3.0
"""
Visualize BEVFormer (post-training) BEV feature maps as energy maps only.

Usage:
    bash tools/dist_visualize_bev.sh <NUM_GPUS>

Or single-process:
    python tools/visualize_bev.py <CONFIG> <CHECKPOINT> --out-dir ../results/bev_visualize_stage2

The script loads a trained BEVFormer (or DiffBEVFormerSeg-style) checkpoint, calls
forward_test(only_bev=True) per sample to obtain the BEV feature [bev_h*bev_w, B, C],
reshapes to [B, C, H, W], and saves a single-panel energy map per sample using the
same aggregation style as projects.bevdiffuser.visualize.bev_visualize.render_bev_triplet.
"""
import argparse
import os
import os.path as osp
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Project plugin paths.
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402
from projects.bevdiffuser.visualize.bev_visualize import (  # noqa: E402
    _aggregate_energy,
    _gaussian_blur_np,
    _percentile_norm_joint,
    bev_extent_from_cfg,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize BEVFormer BEV feature energy maps')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', default='../results/bev_visualize_stage2',
                        help='directory to save energy maps')
    parser.add_argument('--max-samples', type=int, default=-1,
                        help='maximum number of samples to render (-1 = all)')
    parser.add_argument('--agg', default='l1', choices=['l1', 'rms', 'max', 'l1_pos', 'signed_mean'])
    parser.add_argument('--no-whiten', action='store_true', help='disable per-channel whitening')
    parser.add_argument('--smooth-sigma', type=float, default=0.8)
    parser.add_argument('--joint-clip-low', type=float, default=2.0)
    parser.add_argument('--joint-clip-high', type=float, default=98.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--cmap', default='viridis')
    parser.add_argument('--interp', default='bilinear')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--figsize', type=float, nargs=2, default=(5.0, 5.0))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def render_energy_only(bev_bchw, b, out_file, *,
                       agg, whiten, smooth_sigma, joint_clip, gamma,
                       cmap, interp, dpi, figsize, bev_extent, title=None):
    """Render a single-panel energy map for one BEV feature [B,C,H,W]."""
    feat = bev_bchw[b].detach().cpu().float().numpy()  # [C,H,W]
    C, H, W = feat.shape
    flat = feat.reshape(C, H * W).T  # [N,C]

    e = _aggregate_energy(flat, agg=agg, whiten=whiten).reshape(H, W)
    e = _gaussian_blur_np(e, sigma=smooth_sigma)
    # Single-image percentile normalization (joint clip applied to itself).
    e, _ = _percentile_norm_joint(e, e, p_low=joint_clip[0], p_high=joint_clip[1])
    if gamma != 1.0:
        e = np.power(e, gamma).astype(np.float32)

    fig, ax = plt.subplots(1, 1, figsize=tuple(figsize), dpi=dpi)
    ax.imshow(e, cmap=cmap, vmin=0.0, vmax=1.0, interpolation=interp,
              extent=bev_extent, origin='lower')
    ax.set_aspect('equal')
    if bev_extent is not None:
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis('off')

    Path(osp.dirname(out_file)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    plt.close(fig)


def _safe_token(s, fallback):
    if not s:
        return fallback
    s = str(s)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s)[:80]


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Import project plugin modules so the registry is populated.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, 'plugin_dir', osp.dirname(args.config))
        _module_path = '.'.join(osp.dirname(plugin_dir).split('/'))
        importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None

    # Test dataset.
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    dataset_default_args = dict(
        pc_range=cfg.point_cloud_range,
        use_3d_bbox=cfg.use_3d_bbox,
        num_classes=cfg.num_classes,
        num_bboxes=cfg.num_bboxes,
    )
    dataset = build_dataset(cfg.data.test, default_args=dataset_default_args)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
    else:
        model = MMDataParallel(model.cuda(), device_ids=[0])

    bev_h = cfg.bev_h_
    bev_w = cfg.bev_w_
    embed_dim = cfg._dim_
    bev_extent = bev_extent_from_cfg(cfg)

    out_dir = args.out_dir
    rank, world_size = get_dist_info()
    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)

    model.eval()
    rendered = 0
    for step, batch in enumerate(data_loader):
        with torch.no_grad():
            bev = model(return_loss=False, only_bev=True, **batch).detach()

        # bev shape can be [bev_h*bev_w, B, C] or [B, bev_h*bev_w, C]; both reshape OK for B=1.
        bev = bev.reshape(-1, bev_h, bev_w, embed_dim).permute(0, 3, 1, 2).contiguous()

        img_metas = batch['img_metas'][0].data[0]
        sample_token = img_metas[0].get('sample_idx') or img_metas[0].get('sample_token')
        global_idx = step * world_size + rank
        fname = f'{global_idx:06d}_{_safe_token(sample_token, f"sample_{global_idx}")}.png'
        out_file = osp.join(out_dir, fname)

        render_energy_only(
            bev, b=0, out_file=out_file,
            agg=args.agg, whiten=not args.no_whiten,
            smooth_sigma=args.smooth_sigma,
            joint_clip=(args.joint_clip_low, args.joint_clip_high),
            gamma=args.gamma,
            cmap=args.cmap, interp=args.interp,
            dpi=args.dpi, figsize=args.figsize,
            bev_extent=bev_extent,
            title=f'idx {global_idx} | {sample_token or ""}',
        )

        rendered += 1
        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

        if args.max_samples > 0 and rendered >= args.max_samples:
            break

    if rank == 0:
        print(f'\n[visualize_bev] saved {rendered} energy maps to {out_dir}')


if __name__ == '__main__':
    main()
