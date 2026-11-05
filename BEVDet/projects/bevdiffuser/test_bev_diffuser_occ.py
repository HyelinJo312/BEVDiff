"""Evaluate a GT-layout Stage-1 BEVDiffuser checkpoint on Occ3D."""

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
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.utils import setup_multi_processes
from mmdet3d.datasets import build_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

from projects.bevdiffuser.model_utils import build_unet
from projects.mmdet3d_plugin.bevformer.detectors.bevformer_occ import BEVFormerOcc
from projects.bevdiffuser.occ_diffusion_utils import (
    decode_occ_bev,
    get_occ_model,
    load_diffusion_weights,
    preserve_occ_state,
)
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler
from projects.bevdiffuser.train_bev_diffuser_occ import (
    _batch_tensor,
    _bev_tokens_to_map,
    _current_img_metas,
    _layout_condition,
    _resolve_checkpoint,
    _unwrap,
)
from projects.mmdet3d_plugin.bevformer.apis.test import collect_results_cpu
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def _unconditional(condition, unet):
    encoder = _unwrap(unet).layout_encoder
    result = {name: value.clone() for name, value in condition.items()}
    result['obj_class'].fill_(encoder.num_classes_for_layout_object - 1)
    result['obj_class'][:, 0] = encoder.num_classes_for_layout_object - 2
    result['obj_bbox'].zero_()
    result['obj_bbox'][:, 0] = result['obj_bbox'].new_tensor(
        [0, 0, 0, 1, 1, 1, 0, 0, 0])
    result['is_valid_obj'].zero_()
    result['is_valid_obj'][:, 0] = 1.0
    return result


@torch.no_grad()
def evaluate_occ(unet, bev_model, scheduler, dataset, dataloader, cfg,
                 output_dir, noise_timesteps=5, denoise_timesteps=5,
                 num_inference_steps=5, guidance_scale=2.0):
    rank, world_size = get_dist_info()
    device = next(unet.parameters()).device
    core = _unwrap(bev_model)
    if not isinstance(core, BEVFormerOcc):
        raise TypeError('layout_base_occ.py must build BEVFormerOcc')

    diffusion = copy.deepcopy(scheduler)
    if noise_timesteps < 0 or noise_timesteps >= len(diffusion.alphas_cumprod):
        raise ValueError('noise_timesteps is outside the scheduler range')
    if denoise_timesteps:
        if not 0 < num_inference_steps <= denoise_timesteps:
            raise ValueError('Invalid denoising schedule')
        diffusion.register_to_config(num_train_timesteps=denoise_timesteps)
        diffusion.set_timesteps(num_inference_steps=num_inference_steps)

    results = []
    unet.eval()
    with preserve_occ_state(bev_model, reset_temporal=True):
        progress = mmcv.ProgressBar(len(dataset)) if rank == 0 else None
        for batch in dataloader:
            clean_bev = bev_model(
                return_loss=False, only_bev=True, **batch)
            latents = _bev_tokens_to_map(clean_bev, cfg)
            if noise_timesteps:
                timestep = torch.full(
                    (latents.shape[0],), noise_timesteps,
                    device=device, dtype=torch.long)
                latents = diffusion.add_noise(
                    latents, torch.randn_like(latents), timestep)

            if denoise_timesteps:
                condition = _layout_condition(
                    batch, device, unet, uncond_probability=0.0)
                unconditional = _unconditional(condition, unet)
                for timestep in diffusion.timesteps:
                    timestep_batch = torch.full(
                        (latents.shape[0],), int(timestep),
                        device=device, dtype=torch.long)
                    prediction_uncond = unet(
                        latents, timestep_batch, **unconditional)[0]
                    prediction_cond = unet(
                        latents, timestep_batch, **condition)[0]
                    prediction = prediction_uncond + guidance_scale * (
                        prediction_cond - prediction_uncond)
                    latents = diffusion.step(
                        prediction, timestep, latents,
                        return_dict=False)[0]

            img_metas = _current_img_metas(batch)
            outputs = decode_occ_bev(core, latents)
            predictions = core.pts_bbox_head.get_occ(outputs, img_metas)
            results.extend(
                dict(occ_pred=prediction.cpu().numpy().astype(np.uint8))
                for prediction in predictions)
            if progress is not None:
                progress.update(len(predictions) * world_size)

    if world_size > 1:
        results = collect_results_cpu(
            results, len(dataset),
            tmpdir=os.path.join(output_dir, '.dist_test'))
    if rank != 0:
        return None

    eval_kwargs = copy.deepcopy(cfg.get('evaluation', {}))
    for key in ('interval', 'pipeline', 'by_epoch', 'tmpdir',
                'gpu_collect', 'save_best', 'rule', 'start',
                'dynamic_intervals'):
        eval_kwargs.pop(key, None)
    metrics = dataset.evaluate(results[:len(dataset)], **eval_kwargs)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'),
              'w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2, allow_nan=True)
    return metrics


def test():
    args = parse_args()
    cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _resolve_checkpoint(args, cfg)
    setup_multi_processes(cfg)
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    if args.output_dir is None:
        args.output_dir = os.path.join(
            '../results/occ3d_eval',
            os.path.splitext(os.path.basename(args.bev_config))[0],
            os.path.basename(os.path.normpath(args.checkpoint_dir)))
    if rank == 0:
        mmcv.mkdir_or_exist(args.output_dir)

    bev_model = get_occ_model(args, cfg)
    bev_model.requires_grad_(False)
    occ_checkpoint = args.occ_checkpoint
    default_occ = os.path.join(args.checkpoint_dir, 'occ_model.pth')
    if occ_checkpoint is None and os.path.isfile(default_occ):
        occ_checkpoint = default_occ
    if occ_checkpoint:
        load_checkpoint(
            bev_model, occ_checkpoint, map_location='cpu', strict=True)

    unet = build_unet(cfg.unet)
    load_diffusion_weights(unet, args.checkpoint_dir)
    device = next(bev_model.parameters()).device
    unet.to(device=device, dtype=torch.float32)
    unet.requires_grad_(False)

    scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='scheduler')
    scheduler.register_to_config(prediction_type=args.prediction_type)

    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.test_mode = True
    dataset_cfg.load_annos = True
    dataset = build_dataset(dataset_cfg, default_args=dict(test_mode=True))
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        seed=args.seed,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    metrics = evaluate_occ(
        unet, bev_model, scheduler, dataset, dataloader, cfg,
        args.output_dir,
        noise_timesteps=args.noise_timesteps,
        denoise_timesteps=args.denoise_timesteps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale)
    if rank == 0:
        print(json.dumps(metrics, indent=2, allow_nan=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate GT-layout BEVDiffuser on Occ3D')
    parser.add_argument('--bev_config', required=True)
    parser.add_argument('--bev_checkpoint', default=None)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--occ_checkpoint', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--pretrained_model_name_or_path',
                        default='stabilityai/stable-diffusion-2-1')
    parser.add_argument('--prediction_type', default='sample',
                        choices=['sample', 'epsilon', 'v_prediction'])
    parser.add_argument('--noise_timesteps', type=int, default=5)
    parser.add_argument('--denoise_timesteps', type=int, default=5)
    parser.add_argument('--num_inference_steps', type=int, default=5)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument('--samples_per_gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch')
    parser.add_argument('--local_rank', '--local-rank',
                        dest='local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot both be specified')
    if args.options:
        warnings.warn('--options is deprecated; use --cfg-options')
        args.cfg_options = args.options
    return args


if __name__ == '__main__':
    test()
