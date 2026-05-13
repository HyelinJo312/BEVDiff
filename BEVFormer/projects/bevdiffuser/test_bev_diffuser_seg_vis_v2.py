# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Unified qualitative-comparison script.

Loads BOTH the baseline BEVDiffuser UNet and the proposed (semantic-guided)
UNet at the same time, runs two independent diffusion reverse processes on
the same scene, and renders a single 4-panel figure:

    [ Original BEV | BEVDiffuser | Ours (Semantic Guided) | LiDAR Top ]

Usage:
    --bev_config_baseline   path to layout_tiny.py            (BEVDiffuser config)
    --bev_config_ours       path to layout_tiny_seg_v4.py     (Ours config)
    --checkpoint_dir_baseline  baseline UNet checkpoint dir
    --checkpoint_dir_ours      ours UNet checkpoint dir
    --baseline_noise_t / --baseline_denoise_t / --baseline_inference_steps
    --ours_noise_t     / --ours_denoise_t     / --ours_inference_steps
'''

import argparse
import copy
import os, sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.logging import get_logger

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet
# NOTE: two distinct UNet classes share the name `LayoutDiffusionUNetModel`.
# We import them under explicit aliases to keep `isinstance` checks unambiguous
# when building the unconditional layout for either model.
from layout_diffusion.layout_diffusion_unet import (
    LayoutDiffusionUNetModel as BaselineUNetModel,
)
from layout_diffusion.layout_seg_diffusion_unet_v4 import (
    LayoutDiffusionUNetModel as OursUNetModel,
)
from projects.bevdiffuser.visualize.bev_visualize import (
    bev_extent_from_cfg,
    render_bev_quad,
    render_bev_pca_quad,
)

logger = get_logger(__name__, log_level="INFO")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint visualization for BEVDiffuser (baseline) vs Ours."
    )

    # ----- two configs (baseline + ours) -----
    parser.add_argument('--bev_config_baseline', default="",
                        help='Path to baseline (BEVDiffuser) bev config '
                             '(e.g. configs/bevdiffuser/layout_tiny.py)')
    parser.add_argument('--bev_config_ours', default="",
                        help='Path to ours (semantic-guided) bev config '
                             '(e.g. configs/bevdiffuser/layout_tiny_seg_v4.py)')

    # BEV (detector) checkpoint, shared across both pipelines
    parser.add_argument('--bev_checkpoint', default="",
                        help='Shared BEVFormer detector checkpoint (.pth).')

    # ----- two UNet checkpoint dirs -----
    parser.add_argument('--checkpoint_dir_baseline', type=str, default="",
                        help='UNet checkpoint dir for the BEVDiffuser baseline.')
    parser.add_argument('--checkpoint_dir_ours', type=str, default="",
                        help='UNet checkpoint dir for the proposed model.')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str, default="stabilityai/stable-diffusion-2-1",
        choices=["CompVis/stable-diffusion-v1-4",
                 "stabilityai/stable-diffusion-2-1"],
    )
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--use_classifier_guidence", action='store_true')

    # ----- INDEPENDENT sampling hyperparameters per model -----
    # Baseline
    parser.add_argument('--baseline_noise_t', type=int, default=0,
                        help='noise_timesteps for baseline (BEVDiffuser).')
    parser.add_argument('--baseline_denoise_t', type=int, default=5,
                        help='denoise_timesteps for baseline.')
    parser.add_argument('--baseline_inference_steps', type=int, default=5,
                        help='num_inference_steps for baseline.')
    # Ours
    parser.add_argument('--ours_noise_t', type=int, default=0,
                        help='noise_timesteps for ours.')
    parser.add_argument('--ours_denoise_t', type=int, default=5,
                        help='denoise_timesteps for ours.')
    parser.add_argument('--ours_inference_steps', type=int, default=5,
                        help='num_inference_steps for ours.')

    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale (shared).')

    parser.add_argument('--vis_layout', type=str, default='1x4',
                        choices=['1x4', '2x2'])
    parser.add_argument('--vis_mode', type=str, default='pca',
                        choices=['activation', 'pca'],
                        help="'activation': channel-aggregated energy (gray/viridis); "
                             "'pca': DINOv2-style joint RGB-PCA across all 3 BEV maps.")
    parser.add_argument('--out_subdir', type=str, default='compare_quad',
                        help='subdirectory under save_path/visualize.')
    parser.add_argument('--vis_surround', action='store_true',
                        help='If set, also save the 6-camera surround view as a '
                             '2x3 grid (no gaps between images) per step.')
    parser.add_argument('--surround_width', type=int, default=2400,
                        help='Total pixel width of the surround-view 2x3 grid image.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_layout_cond(batch, device):
    """Pull layout_obj_* tensors out of a batch (dataset-config dependent)."""
    cond = {}
    if 'layout_obj_classes' in batch:
        cond['obj_class'] = torch.stack(batch['layout_obj_classes'].data[0])
    if 'layout_obj_bboxes' in batch:
        cond['obj_bbox'] = torch.stack(batch['layout_obj_bboxes'].data[0])
    if 'layout_obj_is_valid' in batch:
        cond['is_valid_obj'] = torch.stack(batch['layout_obj_is_valid'].data[0])
    if 'layout_obj_names' in batch:
        cond['obj_name'] = torch.stack(batch['layout_obj_names'].data[0])
    for k, v in cond.items():
        if isinstance(v, torch.Tensor):
            cond[k] = v.to(device)
    return cond


def _make_layout_uncond(unet, cond, batch, device):
    """Build the CFG-uncond layout for a given UNet (matches its layout_encoder)."""
    uncond = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in cond.items()}
    enc = getattr(unet, 'layout_encoder', None)
    if enc is None:
        return uncond
    used = enc.used_condition_types
    if 'obj_class' in used and 'obj_class' in uncond:
        uncond['obj_class'] = torch.ones_like(uncond['obj_class']).fill_(
            enc.num_classes_for_layout_object - 1
        )
        uncond['obj_class'][:, 0] = enc.num_classes_for_layout_object - 2
    if 'obj_name' in used and 'default_obj_names' in batch:
        uncond['obj_name'] = torch.stack(batch['default_obj_names'].data[0]).to(device)
    if 'obj_bbox' in used and 'obj_bbox' in uncond:
        uncond['obj_bbox'] = torch.zeros_like(uncond['obj_bbox'])
        if getattr(enc, 'use_3d_bbox', False):
            uncond['obj_bbox'][:, 0] = torch.tensor(
                [0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.float32,
                device=uncond['obj_bbox'].device,
            )
        else:
            uncond['obj_bbox'][:, 0] = torch.tensor(
                [0, 0, 1, 1], dtype=torch.float32,
                device=uncond['obj_bbox'].device,
            )
    if 'is_valid_obj' in uncond:
        uncond['is_valid_obj'] = torch.zeros_like(uncond['is_valid_obj'])
        uncond['is_valid_obj'][:, 0] = 1.0
    return uncond


def _add_noise(scheduler, latents, noise, n_t):
    """Replicate the original add-noise logic, supporting the >1000 'pure noise' branch."""
    if n_t > 1000:
        x = torch.randn_like(latents) * scheduler.init_noise_sigma
        return x
    if n_t > 0:
        n_t_tensor = torch.tensor(n_t).long()
        return scheduler.add_noise(latents, noise, n_t_tensor)
    return latents


def _denoise_baseline(unet, scheduler, latents, batch, device,
                      denoise_t, n_inf, cfg_scale, classifier_grad_fn=None):
    """Run reverse process for the BASELINE UNet (no seg_cond, no img_metas)."""
    if denoise_t <= 0:
        return latents
    cond = _make_layout_cond(batch, device)
    uncond = _make_layout_uncond(unet, cond, batch, device)

    scheduler.config.num_train_timesteps = denoise_t
    scheduler.set_timesteps(num_inference_steps=n_inf)

    for t in scheduler.timesteps:
        t_batch = torch.tensor([t] * latents.shape[0], device=device)
        # Baseline UNet signature: (x, timesteps, obj_class, obj_bbox, ..., **kwargs)
        noise_pred_uncond = unet(latents, t_batch, **uncond)[0]
        noise_pred_cond = unet(latents, t_batch, **cond)[0]
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        cls_grad = classifier_grad_fn(latents) if classifier_grad_fn is not None else None
        latents = scheduler.step( noise_pred, t, latents, return_dict=False, classifier_gradient=cls_grad)[0]
    return latents


def _denoise_ours(unet, scheduler, latents, batch, img_metas, seg_cond, depth_maps,
                  device, denoise_t, n_inf, cfg_scale, classifier_grad_fn=None):
    """Run reverse process for the OURS UNet (seg-conditioned + img_metas)."""
    if denoise_t <= 0:
        return latents
    cond = _make_layout_cond(batch, device)
    uncond = _make_layout_uncond(unet, cond, batch, device)
    seg_uncond = torch.zeros_like(seg_cond)

    scheduler.config.num_train_timesteps = denoise_t
    scheduler.set_timesteps(num_inference_steps=n_inf)

    for t in scheduler.timesteps:
        t_batch = torch.tensor([t] * latents.shape[0], device=device)
        # Ours UNet signature: (x, timesteps, img_metas, seg_cond, ..., depth_maps, **kwargs)
        noise_pred_uncond = unet(latents, t_batch, img_metas, seg_uncond,depth_maps=depth_maps, **uncond)[0]
        noise_pred_cond = unet(latents, t_batch, img_metas, seg_cond, depth_maps=depth_maps, **cond)[0]
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        cls_grad = classifier_grad_fn(latents) if classifier_grad_fn is not None else None
        latents = scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=cls_grad)[0]
    return latents


def _bev_default_args(cfg):
    return {
        'pc_range': cfg.point_cloud_range,
        'use_3d_bbox': cfg.use_3d_bbox,
        'num_classes': cfg.num_classes,
        'num_bboxes': cfg.num_bboxes,
    }


# --------------------------------------------------------------------------- #
# Surround-view (6 cameras) helpers
# --------------------------------------------------------------------------- #
_CAM_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]


def _rearrange_cam_paths(paths):
    """Sort camera paths into [FL, F, FR, BL, B, BR] order."""
    def cam_key(p):
        parts = os.path.normpath(p).split(os.sep)
        cam = None
        try:
            i = parts.index('samples')
            cam = parts[i + 1]
        except Exception:
            pass
        return _CAM_ORDER.index(cam) if cam in _CAM_ORDER else len(_CAM_ORDER)
    paths = list(paths)[:6]
    return sorted(paths, key=cam_key)


def _save_surround_grid(cam_paths, out_file, total_width_px=2400):
    """
    Save 6 camera images as a 2x3 grid with NO gaps between tiles.

    All images are resized to a common (tile_w, tile_h) computed from the first
    image's aspect ratio so the grid is gap-free regardless of source size.
    """
    from PIL import Image
    if len(cam_paths) < 6:
        return
    tile_w = max(1, total_width_px // 3)
    first = Image.open(cam_paths[0])
    aspect = first.height / max(1, first.width)
    tile_h = max(1, int(round(tile_w * aspect)))
    canvas = Image.new('RGB', (tile_w * 3, tile_h * 2))
    for idx, p in enumerate(cam_paths[:6]):
        im = Image.open(p).convert('RGB').resize((tile_w, tile_h), Image.BILINEAR)
        r, c = divmod(idx, 3)
        canvas.paste(im, (c * tile_w, r * tile_h))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    canvas.save(out_file)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def test():
    args = parse_args()

    # ---- 1) Load BOTH configs (do NOT mutate the other after this point) ----
    baseline_cfg = Config.fromfile(args.bev_config_baseline)
    ours_cfg = Config.fromfile(args.bev_config_ours)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    # use ours_cfg.dist_params for dist init (both should be identical for layout_tiny family)
    if args.launcher != 'none':
        init_dist(args.launcher, **ours_cfg.dist_params)

    # ---- 2) TWO scheduler instances so each model's set_timesteps() is independent ----
    noise_scheduler_baseline = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_ours = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler_baseline.register_to_config(prediction_type=args.prediction_type)
        noise_scheduler_ours.register_to_config(prediction_type=args.prediction_type)

    # ---- 3) Build BEV detector ONCE (image -> BEV feature is identical for either pipeline) ----
    # We use the OURS config to build the BEV detector because its dataset emits seg_maps
    # and its img_metas naturally feed the ours UNet. The baseline UNet only consumes
    # `latents` (BEV feature) + layout, so it does not care which dataset produced the BEV.
    args_for_bev = copy.copy(args)
    args_for_bev.bev_config = args.bev_config_ours
    bev_model = get_bev_model(args_for_bev)
    if not args.use_classifier_guidence:
        bev_model.requires_grad_(False)
    bev_model.eval()
    device = bev_model.device

    # ---- 4) Build BOTH UNets (separate weights, separate forward signatures) ----
    # NOTE on parameter injection:
    #   - unet_baseline is built from baseline_cfg.unet (12-class layout encoder).
    #     It is fed layout from the BASELINE dataloader (12-class indexing).
    #   - unet_ours is built from ours_cfg.unet (18-class layout encoder + SegBEVAligner).
    #     It is fed layout AND seg_maps from the OURS dataloader (18-class indexing).
    # Mixing layout tensors across configs would crash the embedding lookup.
    unet_baseline = build_unet(baseline_cfg.unet)
    unet_baseline.from_pretrained(args.checkpoint_dir_baseline, subfolder="unet")
    unet_baseline.to(device, dtype=torch.float32)
    unet_baseline.requires_grad_(False)
    unet_baseline.eval()

    unet_ours = build_unet(ours_cfg.unet)
    unet_ours.from_pretrained(args.checkpoint_dir_ours, subfolder="unet")
    unet_ours.to(device, dtype=torch.float32)
    unet_ours.requires_grad_(False)
    unet_ours.eval()

    # ---- 5) Build TWO datasets / dataloaders, one per config ----
    for cfg_ in (baseline_cfg, ours_cfg):
        cfg_.data.test.test_mode = True
        cfg_.data.test.load_annos = True

    baseline_dataset = build_dataset(
        baseline_cfg.data.test, default_args=_bev_default_args(baseline_cfg)
    )
    ours_dataset = build_dataset(
        ours_cfg.data.test, default_args=_bev_default_args(ours_cfg)
    )

    baseline_loader = build_dataloader(
        baseline_dataset,
        samples_per_gpu=baseline_cfg.data.samples_per_gpu,
        workers_per_gpu=baseline_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=baseline_cfg.data.nonshuffler_sampler,
    )
    ours_loader = build_dataloader(
        ours_dataset,
        samples_per_gpu=ours_cfg.data.samples_per_gpu,
        workers_per_gpu=ours_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=ours_cfg.data.nonshuffler_sampler,
    )

    # Save directly under the top-level results dir (no per-checkpoint nesting).
    save_path = '../../../results'

    evaluate(
        unet_baseline=unet_baseline,
        unet_ours=unet_ours,
        bev_model=bev_model,
        scheduler_baseline=noise_scheduler_baseline,
        scheduler_ours=noise_scheduler_ours,
        baseline_loader=baseline_loader,
        ours_loader=ours_loader,
        baseline_dataset=baseline_dataset,
        ours_dataset=ours_dataset,
        baseline_cfg=baseline_cfg,
        ours_cfg=ours_cfg,
        save_path=save_path,
        baseline_noise_t=args.baseline_noise_t,
        baseline_denoise_t=args.baseline_denoise_t,
        baseline_inference_steps=args.baseline_inference_steps,
        ours_noise_t=args.ours_noise_t,
        ours_denoise_t=args.ours_denoise_t,
        ours_inference_steps=args.ours_inference_steps,
        cfg_scale=args.cfg_scale,
        use_classifier_guidence=args.use_classifier_guidence,
        vis_layout=args.vis_layout,
        vis_mode=args.vis_mode,
        out_subdir=args.out_subdir,
        vis_surround=args.vis_surround,
        surround_width=args.surround_width,
    )


# --------------------------------------------------------------------------- #
# Evaluation / visualization loop
# --------------------------------------------------------------------------- #
def evaluate(
    unet_baseline, unet_ours,
    bev_model,
    scheduler_baseline, scheduler_ours,
    baseline_loader, ours_loader,
    baseline_dataset, ours_dataset,
    baseline_cfg, ours_cfg,
    save_path,
    baseline_noise_t, baseline_denoise_t, baseline_inference_steps,
    ours_noise_t, ours_denoise_t, ours_inference_steps,
    cfg_scale=2.0,
    use_classifier_guidence=False,
    vis_layout='1x4',
    vis_mode='pca',
    out_subdir='compare_quad',
    vis_surround=False,
    surround_width=2400,
):
    device = bev_model.device

    # Optional classifier guidance — only meaningful for the BEV detector branch.
    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, ours_cfg.bev_h_ * ours_cfg.bev_w_, ours_cfg._dim_)
        loss = bev_model(return_loss=False, only_bev=False, given_bev=x_,
                         return_eval_loss=True, **kwargs)
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, ours_cfg.bev_h_, ours_cfg.bev_w_, ours_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient

    # nuScenes for LiDAR top-down view
    ds = getattr(ours_loader, "dataset", ours_dataset)
    nusc = getattr(ds, "nusc", None)
    if nusc is None:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version=getattr(ds, "version", "v1.0-trainval"),
            dataroot=getattr(ds, "data_root", "./data/nuscenes"),
            verbose=False,
        )

    extent = bev_extent_from_cfg(ours_cfg)

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(min(len(baseline_dataset), len(ours_dataset)))
    time.sleep(2)

    out_dir = os.path.join(
        save_path, 'visualize', out_subdir,
        f"b{baseline_noise_t}_{baseline_denoise_t}_{baseline_inference_steps}"
        f"__o{ours_noise_t}_{ours_denoise_t}_{ours_inference_steps}",
    )

    # The two dataloaders are aligned: same val.pkl, shuffle=False, same sampler type.
    for step, (baseline_batch, ours_batch) in enumerate(zip(baseline_loader, ours_loader)):

        # ---- (a) BEV feature once (use ours_batch; img is identical across configs) ----
        latents = bev_model(return_loss=False, only_bev=True, **ours_batch).detach()
        latents = latents.reshape(-1, ours_cfg.bev_h_, ours_cfg.bev_w_, ours_cfg._dim_)
        latents = latents.permute(0, 3, 1, 2)
        original_bev = latents.detach().clone()  # (B,C,H,W)

        img_metas = ours_batch['img_metas'][0].data[0]
        sample_token = img_metas[0]['sample_idx']

        # depth maps (Ours-only; currently unused in the pipeline -> None)
        depth_maps = None
        if 'depth_maps' in ours_batch:
            depth_maps = torch.stack(ours_batch['depth_maps'].data[0], dim=0).to(device)

        # seg_cond from ours_batch
        seg_cond = torch.stack(ours_batch['seg_maps'].data[0], dim=0).to(device)

        # ---- (b) Identical noise tensor, then independent noise scaling per model ----
        # Same eps -> baseline & ours start from the same underlying noise pattern.
        eps = torch.randn_like(original_bev)

        latents_baseline = _add_noise(scheduler_baseline, original_bev, eps, baseline_noise_t)
        latents_ours = _add_noise(scheduler_ours, original_bev, eps, ours_noise_t)

        cls_grad_fn = (
            (lambda x: get_classifier_gradient(x, **ours_batch))
            if use_classifier_guidence else None
        )

        # ---- (c) Two independent reverse processes ----
        denoised_baseline = _denoise_baseline(
            unet=unet_baseline,
            scheduler=scheduler_baseline,
            latents=latents_baseline,
            batch=baseline_batch,
            device=device,
            denoise_t=baseline_denoise_t,
            n_inf=baseline_inference_steps,
            cfg_scale=cfg_scale,
            classifier_grad_fn=cls_grad_fn,
        ).detach().clone()

        denoised_ours = _denoise_ours(
            unet=unet_ours,
            scheduler=scheduler_ours,
            latents=latents_ours,
            batch=ours_batch,
            img_metas=img_metas,
            seg_cond=seg_cond,
            depth_maps=depth_maps,
            device=device,
            denoise_t=ours_denoise_t,
            n_inf=ours_inference_steps,
            cfg_scale=cfg_scale,
            classifier_grad_fn=cls_grad_fn,
        ).detach().clone()

        # ---- (d) 4-way visualization (mode-dependent) ----
        common_kwargs = dict(
            bev_original_bchw=original_bev,
            bev_baseline_bchw=denoised_baseline,
            bev_ours_bchw=denoised_ours,
            b=0,
            nusc=nusc, sample_token=sample_token,
            out_dir=out_dir,
            title=f"step {step} | BEV feature comparison",
            labels=("Original BEV", "BEVDiffuser", "Ours", "LiDAR Top"),
            bev_extent=extent, bev_origin="lower",
            lidar_axes_limit=50.0,
            layout=vis_layout,
            figsize=None, dpi=300, show=False,
        )
        # ---- (d.0) Optional: 2x3 surround-view image grid (no gaps) ----
        if vis_surround:
            img_filenames = img_metas[0].get('filename', None)
            if img_filenames is not None:
                cam_paths = _rearrange_cam_paths(img_filenames)
                surround_file = os.path.join(
                    out_dir, 'surround', f"step{step:06d}_{sample_token}.png"
                )
                _save_surround_grid(cam_paths, surround_file,
                                    total_width_px=surround_width)

        if vis_mode == 'pca':
            # DINOv2-style: ONE PCA fit jointly on all 3 BEV maps so that
            # per-component color is comparable across panels.
            render_bev_pca_quad(
                **common_kwargs,
                pca_whiten=False, pca_clip=(0.5, 99.5), pca_gamma=1.0,
                upsample=3, ssaa=True, blur_sigma=0.8,
                edge_preserve="bilateral", interp="bicubic",
            )
        else:  # 'activation'
            render_bev_quad(
                **common_kwargs,
                agg="l1", whiten=True, smooth_sigma=0.8,
                joint_clip=(2.0, 98.0), gamma=1.0,
                bev_cmap="viridis", bev_interp="bilinear",
                save_format="pdf",
            )

        if rank == 0:
            prog_bar.update()


if __name__ == "__main__":
    test()
