# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Evaluate semantic-only Stage-1 BEV diffusion on Occ3D-nuScenes."""

import argparse
import copy
import json
import os
import sys
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.utils import setup_multi_processes
from mmdet3d.datasets import build_dataset

torch.backends.cudnn.enabled = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

from projects.mmdet3d_plugin.bevformer.apis.test import collect_results_cpu
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.bevdiffuser.model_utils import build_unet_v2
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler
from projects.mmdet3d_plugin.bevformer.detectors.bevformer_occ import BEVFormerOcc
from projects.bevdiffuser.occ_diffusion_utils import (
    decode_occ_bev, get_occ_model, load_diffusion_weights,
    preserve_occ_state,
)


def _unwrap(model):
    """Return the underlying BEV model from an MMCV parallel wrapper."""
    return model.module if hasattr(model, "module") else model


def _bev_tokens_to_map(bev, bev_cfg):
    """Convert BEVFormer tokens to diffusion layout ``[B,C,Y,X]``."""
    if bev.dim() == 4:
        expected = (bev_cfg._dim_, bev_cfg.bev_h_, bev_cfg.bev_w_)
        if tuple(bev.shape[1:]) != expected:
            raise ValueError(f"Unexpected BEV map shape {tuple(bev.shape)}; expected "
                             f"[B,{expected[0]},{expected[1]},{expected[2]}]")
        return bev.contiguous()

    if bev.dim() != 3:
        raise ValueError(f"BEV feature must be 3D or 4D, got {bev.shape}")

    h, w, channels = bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_
    if bev.shape[1:] == (h * w, channels):
        tokens = bev
    elif bev.shape[0] == h * w and bev.shape[2] == channels:
        tokens = bev.permute(1, 0, 2)
    else:
        raise ValueError(f"Cannot infer BEV token axis from {tuple(bev.shape)} for "
                         f"H={h}, W={w}, C={channels}")
    return tokens.reshape(tokens.shape[0], h, w, channels).permute(0, 3, 1, 2).contiguous()


def _container_data(value):
    """Extract one-rank data from an MMCV container/test-aug wrapper."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Only single-scale, non-flipped evaluation is supported")
        value = value[0]
    if isinstance(value, DataContainer):
        value = value.data[0]
    return value


def _batch_tensor(batch, key, device):
    value = _container_data(batch[key])
    if isinstance(value, (list, tuple)):
        value = torch.stack(value, dim=0)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must contain tensors, got {type(value)}")
    return value.to(device=device, non_blocking=True)


def _current_img_metas(batch):
    """Extract current-frame metadata from test wrappers."""
    metas = _container_data(batch["img_metas"])
    current = []
    for sample_metas in metas:
        if (isinstance(sample_metas, dict) and sample_metas and all(isinstance(key, int) for key in sample_metas)):
            sample_metas = sample_metas[max(sample_metas)]
        current.append(sample_metas)
    return current


def _condition_batch(batch, device):
    if 'semantic_bev_probabilities' in batch:
        probability = _batch_tensor(batch, 'semantic_bev_probabilities', device)
        return None, None, dict(semantic_prob=probability)
    seg_maps = _batch_tensor(batch, 'seg_maps', device)
    depth_maps = _batch_tensor(batch, 'depth_maps', device) if 'depth_maps' in batch else None
    return seg_maps, depth_maps, {}


def _reset_temporal_state(bev_model):
    inner = _unwrap(bev_model)
    if hasattr(inner, "prev_frame_info"):
        inner.prev_frame_info["prev_bev"] = None
        inner.prev_frame_info["scene_token"] = None
        inner.prev_frame_info["prev_pos"] = 0
        inner.prev_frame_info["prev_angle"] = 0


@torch.no_grad()
def evaluate_occ(unet, bev_model, noise_scheduler, dataset, dataloader, bev_cfg, output_dir,
                 noise_timesteps=5, denoise_timesteps=5, num_inference_steps=5, guidance_scale=2.0):
    """Decode denoised native BEV with the baseline head; use its camera-mask mIoU."""
    rank, world_size = get_dist_info()
    device = next(unet.parameters()).device
    core = _unwrap(bev_model)
    scheduler = copy.deepcopy(noise_scheduler)
    if not isinstance(core, BEVFormerOcc):
        raise TypeError('Use the native occupancy Stage 1 config')
    if not 0 <= noise_timesteps < len(scheduler.alphas_cumprod):
        raise ValueError('Invalid noise timestep')
    if denoise_timesteps > 0:
        if not 0 < num_inference_steps <= denoise_timesteps <= len(scheduler.alphas_cumprod):
            raise ValueError('Invalid denoising schedule')
        scheduler.register_to_config(num_train_timesteps=denoise_timesteps)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        if int(scheduler.timesteps.max()) >= len(scheduler.alphas_cumprod):
            raise ValueError('Denoising timestep exceeds the trained schedule')
    results = []
    modes = {module: module.training for module in unet.modules()}
    unet.eval()
    try:
        with preserve_occ_state(bev_model, reset_temporal=True):
            progress = mmcv.ProgressBar(len(dataset)) if rank == 0 else None
            for batch in dataloader:
                bev = bev_model(return_loss=False, only_bev=True, **batch)
                latents = _bev_tokens_to_map(bev, bev_cfg)
                img_metas = _current_img_metas(batch)
                if noise_timesteps:
                    ts = torch.full((latents.shape[0],), noise_timesteps, device=device, dtype=torch.long)
                    latents = scheduler.add_noise(latents, torch.randn_like(latents), ts)
                if denoise_timesteps:
                    seg, depth, cond_kwargs = _condition_batch(batch, device)
                    if cond_kwargs:
                        uncond_kwargs = dict(semantic_prob=torch.zeros_like(cond_kwargs['semantic_prob']))
                        uncond_seg = None
                    else:
                        uncond_kwargs, uncond_seg = {}, torch.zeros_like(seg)
                    encoder = _unwrap(unet)
                    condition = encoder.encode_seg(seg, img_metas, depth_maps=depth, **cond_kwargs)
                    unconditional = encoder.encode_seg(uncond_seg, img_metas, depth_maps=depth, **uncond_kwargs)
                    for timestep in scheduler.timesteps:
                        ts = torch.full((latents.shape[0],), int(timestep), device=device, dtype=torch.long)
                        uncond = unet(latents, ts, img_metas, None, seg_bev_maps=unconditional)
                        cond = unet(latents, ts, img_metas, None, seg_bev_maps=condition)
                        prediction = uncond + guidance_scale * (cond - uncond)
                        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                outputs = decode_occ_bev(core, latents)
                predictions = core.pts_bbox_head.get_occ(outputs, img_metas)
                results.extend(dict(occ_pred=pred.cpu().numpy().astype(np.uint8)) for pred in predictions)
                if progress is not None:
                    progress.update(len(predictions) * world_size)
    finally:
        for module, training in modes.items():
            module.training = training
    if world_size > 1:
        results = collect_results_cpu(results, len(dataset), tmpdir=os.path.join(output_dir, '.dist_test'))
    if rank != 0:
        return None
    eval_kwargs = copy.deepcopy(bev_cfg.get('evaluation', {}))
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'pipeline', 'by_epoch', 'dynamic_intervals'):
        eval_kwargs.pop(key, None)
    metrics = dataset.evaluate(results[:len(dataset)], **eval_kwargs)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as stream:
        json.dump(metrics, stream, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate semantic-only Stage-1 BEV diffusion on Occ3D")
    parser.add_argument("--bev_config", required=True)
    parser.add_argument("--bev_checkpoint", required=True,
                        help="Frozen BEVFormerOcc epoch checkpoint")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Stage-1 checkpoint directory containing the unet subdirectory")
    parser.add_argument(
        "--occ_checkpoint", default=None, help=("Optional trained occupancy-model checkpoint. If omitted, "
                                                "checkpoint_dir/occ_model.pth is loaded when present."))
    parser.add_argument("--output_dir", default=None,
                        help="Output root; a sampler-settings subdirectory is appended automatically")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="pytorch")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1",
                        choices=[
                            "CompVis/stable-diffusion-v1-4",
                            "stabilityai/stable-diffusion-2-1",
                        ])
    parser.add_argument("--prediction_type", type=str, default="sample", choices=["epsilon", "sample", "v_prediction"])
    parser.add_argument("--noise_timesteps", type=int, default=5)
    parser.add_argument("--denoise_timesteps", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--samples_per_gpu", type=int, default=None)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--options", nargs="+", action=DictAction, help="Deprecated alias for --cfg-options")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError("--options and --cfg-options cannot both be specified")
    if args.options:
        warnings.warn("--options is deprecated; use --cfg-options instead")
        args.cfg_options = args.options
    if args.noise_timesteps < 0 or args.denoise_timesteps < 0:
        raise ValueError("Diffusion timestep counts must be non-negative")
    if args.denoise_timesteps > 0 and args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be positive when denoising")
    return args


def test():
    args = parse_args()
    bev_cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        bev_cfg.merge_from_dict(args.cfg_options)
    if "unet" not in bev_cfg:
        raise KeyError("The evaluation config must define 'unet'. Use "
                       "projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_occ.py "
                       "instead of the plain Occ3D training config.")
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {args.checkpoint_dir}")

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    setup_multi_processes(bev_cfg)
    if args.launcher != "none":
        init_dist(args.launcher, **bev_cfg.dist_params)

    rank, _ = get_dist_info()
    if args.output_dir is None:
        config_name = os.path.splitext(os.path.basename(args.bev_config))[0]
        checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
        args.output_dir = os.path.join("../results/occ3d_eval", config_name, checkpoint_name)
    sampler_name = (
        f"noise_t{args.noise_timesteps}_denoise_t{args.denoise_timesteps}"
        f"_steps{args.num_inference_steps}_cfg{args.guidance_scale:g}"
    )
    args.output_dir = os.path.join(args.output_dir, sampler_name)
    if rank == 0:
        mmcv.mkdir_or_exist(args.output_dir)
        print(f"Evaluation output directory: {args.output_dir}")

    noise_scheduler = DDIMGuidedScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    bev_model = get_occ_model(args, bev_cfg)
    bev_model.requires_grad_(False)
    bev_model.eval()

    occ_checkpoint = args.occ_checkpoint
    checkpoint_occ = os.path.join(args.checkpoint_dir, "occ_model.pth")
    if occ_checkpoint is None and os.path.isfile(checkpoint_occ):
        occ_checkpoint = checkpoint_occ
    if occ_checkpoint is not None:
        if not os.path.isfile(occ_checkpoint):
            raise FileNotFoundError(f"Occupancy checkpoint not found: {occ_checkpoint}")
        load_checkpoint(bev_model, occ_checkpoint, map_location="cpu", strict=True)

    unet = build_unet_v2(bev_cfg.unet)
    load_diffusion_weights(unet, args.checkpoint_dir)
    device = next(bev_model.parameters()).device
    unet.to(device=device, dtype=torch.float32)
    unet.requires_grad_(False)
    unet.eval()

    test_cfg = bev_cfg.data.test.copy()
    test_cfg.test_mode = True
    dataset = build_dataset(test_cfg, default_args=dict(test_mode=True))
    samples_per_gpu = args.samples_per_gpu
    if samples_per_gpu is None:
        samples_per_gpu = test_cfg.get("samples_per_gpu", 1)

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != "none"),
        shuffle=False,
        seed=args.seed,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    metrics = evaluate_occ(
        unet=unet,
        bev_model=bev_model,
        noise_scheduler=noise_scheduler,
        dataset=dataset,
        dataloader=dataloader,
        bev_cfg=bev_cfg,
        output_dir=args.output_dir,
        noise_timesteps=args.noise_timesteps,
        denoise_timesteps=args.denoise_timesteps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    if rank == 0:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, allow_nan=True)
        print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    test()
