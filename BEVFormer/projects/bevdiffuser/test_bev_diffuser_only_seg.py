# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Following code is adapted from
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
from safetensors.torch import load_file
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
import os, sys
import time

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
# from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel

import mmcv
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed
from mmdet.utils import setup_multi_processes

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet_v2

logger = get_logger(__name__, log_level="INFO")

def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument('--bev_config',
                        default="",
                        help='test config file path')

    parser.add_argument('--bev_checkpoint',
                        default="",
                        help='checkpoint file')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        choices=[
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1"
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="The checkpoint directory of unet.",
    )


    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--prediction_type",
        type=str,
        default='sample',
        choices=["sample", "epsilon", "v_prediction"],
        help="Prediction target used in Stage 1 training; default sample matches train_only_seg_cache.sh.",
    )

    parser.add_argument(
        "--use_classifier_guidence",
        action='store_true',
        help="whether to use classifier guidence",
    )

    parser.add_argument(
        '--noise_timesteps',
        type=int,
        default=0,
        help='The number of timesteps to add noise.')

    parser.add_argument(
        '--denoise_timesteps',
        type=int,
        default=5,
        help='The number of timesteps to denoise.')

    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=5,
        help='The number of diffusion steps to run the unet.')

    parser.add_argument(
        '--cfg_scale',
        type=float,
        default=2,
        help='Classifier-free guidance scale.')

    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')


    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    setup_multi_processes(bev_cfg)
    if args.launcher != 'none':
        init_dist(args.launcher, **bev_cfg.dist_params)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    bev_model = get_bev_model(args).cuda()
    # Guidance needs gradients with respect to BEV inputs, not model weights.
    bev_model.requires_grad_(False)
    bev_model.eval()

    unet = build_unet_v2(bev_cfg.unet)
    checkpoint = Path(args.checkpoint_dir)
    candidates = ([checkpoint] if checkpoint.is_file() else
                  [checkpoint / SAFETENSORS_WEIGHTS_NAME,
                   checkpoint / 'unet' / SAFETENSORS_WEIGHTS_NAME])
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(f'No UNet safetensors checkpoint found at {args.checkpoint_dir}')
    unet.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)
    unet.to(next(bev_model.parameters()).device, dtype=torch.float32)
    unet.requires_grad_(False)
    unet.eval()

    from projects.bevdiffuser import data_utils  # Register Metric3D/cache datasets.
    bev_cfg.data.test.test_mode = True
    bev_cfg.data.test.load_annos = True
    dataset = build_dataset(bev_cfg.data.test,
                            default_args={
                                          'pc_range': bev_cfg.point_cloud_range,
                                          'use_3d_bbox': bev_cfg.use_3d_bbox,
                                          'num_classes': bev_cfg.num_classes,
                                          'num_bboxes': bev_cfg.num_bboxes,
                                    })
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # BEVFormer temporal evaluation handles one frame per rank.
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    save_path = os.path.join('../../test', args.bev_config.split('/')[-1].split('.')[-2], args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])

    evaluate(unet=unet,
             bev_model=bev_model,
             noise_scheduler=noise_scheduler,
             dataset=dataset,
             dataloader=dataloader,
             bev_cfg=bev_cfg,
             eval=args.eval,
             save_path=save_path,
             noise_timesteps=args.noise_timesteps,
             denoise_timesteps=args.denoise_timesteps,
             num_inference_steps=args.num_inference_steps,
             cfg_scale=args.cfg_scale,
             use_classifier_guidence=args.use_classifier_guidence,)


@contextmanager
def _evaluation_state(unet, bev_model):
    """Isolate eval mode and temporal history from an ongoing training run."""
    modules = {module: module.training for model in (unet, bev_model)
               for module in model.modules()}
    core = getattr(bev_model, 'module', bev_model)
    history = getattr(core, 'prev_frame_info', None)
    try:
        unet.eval()
        bev_model.eval()
        if history is not None:
            core.prev_frame_info = dict(prev_bev=None, scene_token=None,
                                       prev_pos=0, prev_angle=0)
        yield
    finally:
        for module, training in modules.items():
            module.training = training
        if history is not None:
            core.prev_frame_info = history


@torch.no_grad()
def evaluate(unet,
             bev_model,
             noise_scheduler,
             dataset,
             dataloader,
             bev_cfg,
             eval='bbox',
             save_path='',
             noise_timesteps=0,
             denoise_timesteps=0,
             num_inference_steps=0,
             cfg_scale=2.0,
             use_classifier_guidence=False):

    with _evaluation_state(unet, bev_model):
        @torch.enable_grad()
        def get_classifier_gradient(x, **kwargs):
            # Test-time augmentation wraps GT in [DataContainer], while the
            # detection loss expects the batch list inside that container.
            kwargs = dict(kwargs)
            for key in ('gt_bboxes_3d', 'gt_labels_3d'):
                value = kwargs.get(key)
                if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], DataContainer):
                    kwargs[key] = value[0]
            x_ = x.detach().requires_grad_(True)
            x_ = x_.permute(0, 2, 3, 1)
            x_ = x_.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
            loss = bev_model(return_loss=False, only_bev=False, given_bev=x_, return_eval_loss=True, **kwargs)
            gradient = torch.autograd.grad(loss, x_)[0]
            gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
            gradient = gradient.permute(0, 3, 1, 2)
            return gradient

        # Preserve the trained alpha schedule and the caller's scheduler settings.
        noise_scheduler = copy.deepcopy(noise_scheduler)
        training_steps = len(noise_scheduler.alphas_cumprod)
        if noise_timesteps < 0 or denoise_timesteps < 0:
            raise ValueError('Noise and denoise timesteps must be nonnegative')
        if noise_timesteps == training_steps:
            raise ValueError(f'Noise timestep must be below {training_steps}; use a larger value for pure noise')
        if denoise_timesteps > 0:
            if not 0 < num_inference_steps <= denoise_timesteps <= training_steps:
                raise ValueError('Require 0 < inference steps <= denoise timesteps <= training timesteps')
            noise_scheduler.register_to_config(num_train_timesteps=denoise_timesteps)
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            if int(noise_scheduler.timesteps.max()) >= training_steps:
                raise ValueError('Denoising schedule exceeds the trained alpha schedule')

        det_res_path = f"{noise_timesteps}_{denoise_timesteps}_{num_inference_steps}"
        bbox_results = []
        mask_results = []
        have_mask = False

        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.

        for step, batch in enumerate(dataloader):

            latents = bev_model(return_loss=False, only_bev=True, **batch).detach()

            latents = latents.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)

            latents = latents.permute(0, 3, 1, 2)

            img_metas = batch['img_metas'][0].data[0]  # list of B dicts (single frame)

            depth_maps = None
            if 'depth_maps' in batch.keys():
                depth_maps = torch.stack(batch['depth_maps'].data[0], dim=0).to(latents.device)

            def get_segmaps_uncond(segmaps):
                uncond = torch.zeros_like(segmaps)
                return uncond

            if noise_timesteps > 0:
                if noise_timesteps > training_steps:
                    latents = torch.randn_like(latents)
                    latents = latents * noise_scheduler.init_noise_sigma
                else:
                    noise = torch.randn_like(latents)
                    noise_t = torch.full((latents.shape[0],), noise_timesteps,
                                         dtype=torch.long, device=latents.device)
                    latents = noise_scheduler.add_noise(latents, noise, noise_t)

            if denoise_timesteps > 0:
                # # DDIM
                cond_kwargs, uncond_kwargs = {}, {}
                if 'semantic_bev_probabilities' in batch:
                    semantic_prob = torch.stack(batch['semantic_bev_probabilities'].data[0], dim=0).to(latents.device)
                    cond_kwargs['semantic_prob'] = semantic_prob
                    uncond_kwargs['semantic_prob'] = torch.zeros_like(semantic_prob)
                    seg_cond = seg_uncond = None
                else:
                    seg_cond = torch.stack(batch['seg_maps'].data[0], dim=0).to(latents.device)
                    seg_uncond = get_segmaps_uncond(seg_cond)

                # The semantic encoder is independent of timestep. Encode each CFG
                # branch once; the UNet still recomputes its timestep/global gates.
                encoder = getattr(unet, 'module', unet)
                seg_bev_cond = encoder.encode_seg(seg_cond, img_metas, depth_maps=depth_maps, **cond_kwargs)
                seg_bev_uncond = encoder.encode_seg(seg_uncond, img_metas, depth_maps=depth_maps, **uncond_kwargs)

                for t in noise_scheduler.timesteps:
                    t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)
                    noise_pred_uncond = unet(latents, t_batch, img_metas, None,
                                             seg_bev_maps=seg_bev_uncond)
                    noise_pred_cond = unet(latents, t_batch, img_metas, None,
                                           seg_bev_maps=seg_bev_cond)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                    classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
                    latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]

            # get detection results
            latents = latents.permute(0, 2, 3, 1)
            latents = latents.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
            det_result = bev_model(return_loss=False, only_bev=False, given_bev=latents, rescale=True, **batch)

            if isinstance(det_result, dict):
                if 'bbox_results' in det_result.keys():
                    bbox_result = det_result['bbox_results']
                    batch_size = len(det_result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in det_result.keys() and det_result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(det_result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(det_result)
                bbox_results.extend(det_result)

            if rank == 0:
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        if world_size > 1:
            bbox_results = collect_results_cpu(bbox_results, len(dataset),
                                               tmpdir=os.path.join(save_path, '.dist_test'))
            if have_mask:
                mask_results = collect_results_cpu(mask_results, len(dataset),
                                                   tmpdir=os.path.join(save_path, '.dist_test_mask'))
        if not have_mask:
            mask_results = None

        det_results = bbox_results if mask_results is None else {'bbox_results': bbox_results, 'mask_results': mask_results}

        key_score = {}
        if rank == 0:
            eval_kwargs = bev_cfg.get('evaluation', {}).copy()
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'by_epoch', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            if eval is not None:
                eval_kwargs['metric'] = eval
            eval_kwargs['jsonfile_prefix'] = os.path.join(save_path, det_res_path)
            eval_results = dataset.evaluate(det_results, **eval_kwargs)
            for metric, score in eval_results.items():
                if 'mAP' in  metric or 'NDS' in metric:
                    key_score[metric] = score
        return key_score


if __name__ == "__main__":
    test()
