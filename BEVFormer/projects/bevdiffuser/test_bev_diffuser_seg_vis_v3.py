# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

'''
Qualitative BEV-feature comparison for the semantic-only (v5) teacher.

Same 4-panel figure as test_bev_diffuser_seg_vis_v2.py:

    [ Original BEV | BEVDiffuser | Ours (Semantic Guided) | LiDAR Top ]

Differences from v2:
  - "Ours" is the semantic-only UNet (seg_diffusion_unet_v5, built with
    build_unet_v2) conditioned on cached semantic BEV probabilities
    (`semantic_bev_probabilities`); CFG null = zero probabilities, as in training.
    v2 targeted the layout+seg UNet (layout_seg_diffusion_unet_v4) fed raw seg_maps.
  - The ours UNet returns a tensor; the baseline layout UNet returns a tuple.
  - The BEV feature is computed for every val frame (BEVFormer is temporal), but the
    diffusion + rendering only runs on every `--vis_stride`-th frame.
  - The baseline dataset reuses the ours dataset's data_root / ann_file so both
    loaders read the same val split from the same working directory.
'''

import argparse
import copy
import os, sys

import numpy as np
import torch

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet, build_unet_v2
from projects.bevdiffuser.visualize.bev_visualize import (
    bev_extent_from_cfg,
    render_bev_quad,
    render_bev_pca_quad,
)
from test_bev_diffuser_seg_vis_v2 import (
    _make_layout_cond,
    _make_layout_uncond,
    _add_noise,
    _bev_default_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='BEV feature comparison: BEVDiffuser (layout) vs Ours (semantic-only v5).')
    parser.add_argument('--bev_config_baseline', required=True)
    parser.add_argument('--bev_config_ours', required=True)
    parser.add_argument('--bev_checkpoint', required=True,
                        help='Shared frozen BEVFormer checkpoint (source of the original BEV).')
    parser.add_argument('--checkpoint_dir_baseline', required=True)
    parser.add_argument('--checkpoint_dir_ours', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='stabilityai/stable-diffusion-2-1')
    parser.add_argument('--prediction_type', type=str, default='sample')

    parser.add_argument('--baseline_noise_t', type=int, default=5)
    parser.add_argument('--baseline_denoise_t', type=int, default=5)
    parser.add_argument('--baseline_inference_steps', type=int, default=5)
    parser.add_argument('--ours_noise_t', type=int, default=100)
    parser.add_argument('--ours_denoise_t', type=int, default=100)
    parser.add_argument('--ours_inference_steps', type=int, default=5)
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='CFG scale shared by both models (Stage 2 default is 2).')

    parser.add_argument('--vis_stride', type=int, default=50,
                        help='Render every N-th val frame (BEV is still computed for all frames).')
    parser.add_argument('--max_vis', type=int, default=-1, help='Stop after this many figures (-1: no limit).')
    parser.add_argument('--vis_layout', type=str, default='1x4', choices=['1x4', '2x2'])
    parser.add_argument('--vis_mode', type=str, default='pca', choices=['activation', 'pca'])
    parser.add_argument('--save_root', type=str, default='../../../results/visualize')
    parser.add_argument('--out_subdir', type=str, default='bev_compare_onlyseg_v5')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


@torch.no_grad()
def _denoise_baseline(unet, scheduler, latents, batch, device, denoise_t, n_inf, cfg_scale):
    """BEVDiffuser layout UNet: (x, t, **layout) -> tuple; CFG null = empty layout."""
    if denoise_t <= 0:
        return latents
    cond = _make_layout_cond(batch, device)
    uncond = _make_layout_uncond(unet, cond, batch, device)
    scheduler.config.num_train_timesteps = denoise_t
    scheduler.set_timesteps(num_inference_steps=n_inf)
    for t in scheduler.timesteps:
        t_batch = torch.tensor([t] * latents.shape[0], device=device)
        pred_uncond = unet(latents, t_batch, **uncond)[0]
        pred_cond = unet(latents, t_batch, **cond)[0]
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    return latents


@torch.no_grad()
def _denoise_ours(unet, scheduler, latents, semantic_prob, device, denoise_t, n_inf, cfg_scale):
    """Semantic-only v5 UNet: CFG null = zero semantic probabilities (training convention)."""
    if denoise_t <= 0:
        return latents
    seg_cond = unet.seg_aligner.encode_prob(semantic_prob)
    seg_uncond = unet.seg_aligner.encode_prob(torch.zeros_like(semantic_prob))
    scheduler.config.num_train_timesteps = denoise_t
    scheduler.set_timesteps(num_inference_steps=n_inf)
    for t in scheduler.timesteps:
        t_batch = torch.tensor([t] * latents.shape[0], device=device)
        pred_uncond = unet(latents, t_batch, None, seg_bev_maps=seg_uncond)
        pred_cond = unet(latents, t_batch, None, seg_bev_maps=seg_cond)
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    return latents


def _load_unet(builder, cfg, ckpt_dir, device):
    unet = builder(cfg.unet)
    unet.from_pretrained(ckpt_dir, subfolder='unet')
    unet.to(device, dtype=torch.float32)
    unet.requires_grad_(False)
    unet.eval()
    return unet


def main():
    args = parse_args()
    baseline_cfg = Config.fromfile(args.bev_config_baseline)
    ours_cfg = Config.fromfile(args.bev_config_ours)
    set_random_seed(args.seed, deterministic=False)

    schedulers = {}
    for name in ('baseline', 'ours'):
        s = DDIMGuidedScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
        s.register_to_config(prediction_type=args.prediction_type)
        schedulers[name] = s

    args_for_bev = copy.copy(args)
    args_for_bev.bev_config = args.bev_config_ours
    bev_model = get_bev_model(args_for_bev)
    bev_model.requires_grad_(False)
    bev_model.eval()
    device = next(bev_model.parameters()).device

    unet_baseline = _load_unet(build_unet, baseline_cfg, args.checkpoint_dir_baseline, device)
    unet_ours = _load_unet(build_unet_v2, ours_cfg, args.checkpoint_dir_ours, device)

    from projects.bevdiffuser import data_utils  # noqa: F401  (registers the cache dataset)
    for cfg_ in (baseline_cfg, ours_cfg):
        cfg_.data.test.test_mode = True
        cfg_.data.test.load_annos = True
    # Same val split from the same working directory for both loaders.
    baseline_cfg.data.test.data_root = ours_cfg.data.test.data_root
    baseline_cfg.data.test.ann_file = ours_cfg.data.test.ann_file

    baseline_dataset = build_dataset(baseline_cfg.data.test, default_args=_bev_default_args(baseline_cfg))
    ours_dataset = build_dataset(ours_cfg.data.test, default_args=_bev_default_args(ours_cfg))
    loaders = [build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=cfg_.data.workers_per_gpu,
                                dist=False, shuffle=False,
                                nonshuffler_sampler=cfg_.data.nonshuffler_sampler)
               for ds, cfg_ in ((baseline_dataset, baseline_cfg), (ours_dataset, ours_cfg))]

    from nuscenes.nuscenes import NuScenes
    nusc = getattr(ours_dataset, 'nusc', None) or NuScenes(
        version=getattr(ours_dataset, 'version', 'v1.0-trainval'),
        dataroot=ours_cfg.data.test.data_root, verbose=False)
    extent = bev_extent_from_cfg(ours_cfg)

    out_dir = os.path.join(
        args.save_root, args.out_subdir,
        f'b{args.baseline_noise_t}_{args.baseline_denoise_t}_{args.baseline_inference_steps}'
        f'__o{args.ours_noise_t}_{args.ours_denoise_t}_{args.ours_inference_steps}_cfg{args.cfg_scale:g}')
    os.makedirs(out_dir, exist_ok=True)

    n_total = min(len(baseline_dataset), len(ours_dataset))
    prog_bar = mmcv.ProgressBar(n_total)
    n_vis = 0
    for step, (baseline_batch, ours_batch) in enumerate(zip(*loaders)):
        # BEVFormer is temporal: run it on every frame so prev_bev stays consistent.
        with torch.no_grad():
            latents = bev_model(return_loss=False, only_bev=True, **ours_batch).detach()
        prog_bar.update()
        if step % args.vis_stride:
            continue

        img_metas = ours_batch['img_metas'][0].data[0]
        sample_token = img_metas[0]['sample_idx']
        base_token = baseline_batch['img_metas'][0].data[0][0]['sample_idx']
        if base_token != sample_token:
            raise RuntimeError(f'Loaders out of sync at step {step}: {base_token} vs {sample_token}')

        original_bev = latents.reshape(-1, ours_cfg.bev_h_, ours_cfg.bev_w_, ours_cfg._dim_) \
                              .permute(0, 3, 1, 2).contiguous()
        semantic_prob = torch.stack(ours_batch['semantic_bev_probabilities'].data[0], 0).to(device).float()

        # One shared noise sample; each model scales it to its own noise level.
        gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + step)
        eps = torch.randn(original_bev.shape, generator=gen, device=device)

        denoised_baseline = _denoise_baseline(
            unet_baseline, schedulers['baseline'],
            _add_noise(schedulers['baseline'], original_bev, eps, args.baseline_noise_t),
            baseline_batch, device, args.baseline_denoise_t, args.baseline_inference_steps, args.cfg_scale)
        denoised_ours = _denoise_ours(
            unet_ours, schedulers['ours'],
            _add_noise(schedulers['ours'], original_bev, eps, args.ours_noise_t),
            semantic_prob, device, args.ours_denoise_t, args.ours_inference_steps, args.cfg_scale)

        common = dict(
            bev_original_bchw=original_bev, bev_baseline_bchw=denoised_baseline,
            bev_ours_bchw=denoised_ours, b=0, nusc=nusc, sample_token=sample_token,
            out_dir=out_dir, title=f'step {step:04d} {sample_token}',
            labels=('Original BEV', 'BEVDiffuser', 'Ours (Semantic Guided)', 'LiDAR Top'),
            bev_extent=extent, bev_origin='lower', lidar_axes_limit=50.0,
            layout=args.vis_layout, figsize=None, dpi=300, show=False)
        if args.vis_mode == 'pca':
            render_bev_pca_quad(**common, pca_whiten=False, pca_clip=(0.5, 99.5), pca_gamma=1.0,
                                upsample=3, ssaa=True, blur_sigma=0.8,
                                edge_preserve='bilateral', interp='bicubic')
        else:
            render_bev_quad(**common, agg='l1', whiten=True, smooth_sigma=0.8,
                            joint_clip=(2.0, 98.0), gamma=1.0, bev_cmap='viridis',
                            bev_interp='bilinear', save_format='png')
        n_vis += 1
        if 0 < args.max_vis <= n_vis:
            break
    print(f'\nSaved {n_vis} figures to {out_dir}')


if __name__ == '__main__':
    main()
