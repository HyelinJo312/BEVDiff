# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Image <-> condition MISMATCH experiment (Experiment 2).

Goal: measure whether the diffusion model relies on the input-image BEV feature
or on the diffusion condition (layout + seg). We feed the image from one sample
and the condition from a *different* (K-step away, different-scene) donor sample,
then evaluate detection performance.

Three modes (`--mismatch_mode`), all evaluated against the *anchor* sample's GT so
that `dataset.evaluate` works in dataset order without any result reordering:

    matched : image = condition = anchor                  (control, single-frame)
    cond_gt : image = DONOR(B), condition = anchor(A=GT)  -> A-GT stays high  => condition shortcut
    img_gt  : image = anchor(B=GT), condition = DONOR(A)  -> B-GT stays high  => image dependence

Note: temporal (video_test_mode) is disabled for all modes (see --disable_temporal)
so the image-BEV depends only on the current frame; otherwise the donor scene jump
would corrupt the temporal state and confound the result.
'''

import argparse
import math
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
from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import collate as mmcv_collate
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet, build_unet_v2
from layout_diffusion.layout_dino_diffusion_unet_v2 import LayoutDiffusionUNetModel
from projects.bevdiffuser.fm_feature import GetDINOV2Feat

logger = get_logger(__name__, log_level="INFO")

def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Image<->condition mismatch experiment.")

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
    )

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training.",
    )

    parser.add_argument(
        "--use_classifier_guidence",
        action='store_true',
        help="whether to use classifier guidence",
    )

    parser.add_argument(
        '--noise_timesteps',
        type=int,
        default=100,
        help='The number of timesteps to add noise.')

    parser.add_argument(
        '--denoise_timesteps',
        type=int,
        default=100,
        help='The number of timesteps to denoise.')

    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=5,
        help='The number of diffusion steps to run the unet.')

    parser.add_argument(
        '--inversion',
        type=bool,
        default=False,
        help='Use inversion process for diffusion sampling.')

    # ---- mismatch-specific args ----
    parser.add_argument(
        '--mismatch_mode',
        type=str,
        default='cond_gt',
        choices=['matched', 'cond_gt', 'img_gt'],
        help="matched: control. cond_gt: image=donor, cond=anchor(GT). "
             "img_gt: image=anchor(GT), cond=donor.")

    parser.add_argument(
        '--donor_offset',
        type=int,
        default=500,
        help='K-step gap to the donor sample (then advanced until a different scene).')

    parser.add_argument(
        '--disable_temporal',
        type=lambda x: str(x).lower() not in ('false', '0', 'no'),
        default=True,
        help='Disable BEVFormer temporal (video_test_mode) so image-BEV is single-frame.')

    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g. "bbox" / "mAP".')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def build_donor_map(dataset, offset):
    """For each index, pick a donor index that is `offset` away and from a
    different scene. Deterministic -> reproducible."""
    N = len(dataset)
    try:
        scene_tokens = [dataset.data_infos[i]['scene_token'] for i in range(N)]
    except Exception:
        scene_tokens = [None] * N  # fallback: no scene info -> rely on offset only

    donor_idx = [0] * N
    for i in range(N):
        j = (i + offset) % N
        guard = 0
        while scene_tokens[j] is not None and scene_tokens[i] is not None \
                and scene_tokens[j] == scene_tokens[i] and guard < N:
            j = (j + 1) % N
            guard += 1
        donor_idx[i] = j
    return donor_idx


def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    if args.launcher != 'none':
        init_dist(args.launcher, **bev_cfg.dist_params)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    bev_model = get_bev_model(args)
    if not args.use_classifier_guidence:
        bev_model.requires_grad_(False)
    bev_model.eval()

    # Disable temporal so the image-BEV depends only on the current frame.
    if args.disable_temporal:
        try:
            bev_model.module.video_test_mode = False
        except AttributeError:
            bev_model.video_test_mode = False

    unet = build_unet(bev_cfg.unet)
    unet.from_pretrained(args.checkpoint_dir, subfolder="unet")
    unet.to(bev_model.device, dtype=torch.float32)
    unet.requires_grad_(False)
    unet.eval()

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
        samples_per_gpu=bev_cfg.data.samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    donor_map = build_donor_map(dataset, args.donor_offset) \
        if args.mismatch_mode != 'matched' else None

    save_path = os.path.join('../../test',
                             args.bev_config.split('/')[-1].split('.')[-2],
                             args.checkpoint_dir.split('/')[-2],
                             args.checkpoint_dir.split('/')[-1] + f'_mismatch_{args.mismatch_mode}')

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
             use_classifier_guidence=args.use_classifier_guidence,
             mismatch_mode=args.mismatch_mode,
             donor_map=donor_map,
             samples_per_gpu=bev_cfg.data.samples_per_gpu,)


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
             use_classifier_guidence=False,
             mismatch_mode='cond_gt',
             donor_map=None,
             samples_per_gpu=1):

    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        loss = bev_model(return_loss=False, only_bev=False, given_bev=x_, return_eval_loss=True, **kwargs)
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient

    def get_condition(cond_batch, device, use_cond=True):
        cond = {}
        if 'layout_obj_classes' in cond_batch:
            cond['obj_class'] = torch.stack(cond_batch['layout_obj_classes'].data[0])
        if 'layout_obj_bboxes' in cond_batch:
            cond['obj_bbox'] = torch.stack(cond_batch['layout_obj_bboxes'].data[0])
        if 'layout_obj_is_valid' in cond_batch:
            cond['is_valid_obj'] = torch.stack(cond_batch['layout_obj_is_valid'].data[0])
        if 'layout_obj_names' in cond_batch:
            cond['obj_name'] = torch.stack(cond_batch['layout_obj_names'].data[0])

        if not use_cond:
            if isinstance(unet, LayoutDiffusionUNetModel):
                if 'obj_class' in unet.layout_encoder.used_condition_types:
                    cond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(unet.layout_encoder.num_classes_for_layout_object - 1)
                    cond['obj_class'][:, 0] = unet.layout_encoder.num_classes_for_layout_object - 2
                if 'obj_name' in unet.layout_encoder.used_condition_types:
                    cond['obj_name'] = torch.stack(cond_batch['default_obj_names'].data[0])
                if 'obj_bbox' in unet.layout_encoder.used_condition_types:
                    cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
                    if unet.layout_encoder.use_3d_bbox:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
                    else:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 1, 1])
                cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                cond['is_valid_obj'][:, 0] = 1.0
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond[key] = value.to(device)
        return cond

    def get_segmaps_uncond(segmaps):
        return torch.zeros_like(segmaps)

    det_res_path = f"{mismatch_mode}_{noise_timesteps}_{denoise_timesteps}_{num_inference_steps}"
    bbox_results = []
    mask_results = []
    have_mask = False

    rank, world_size = get_dist_info()
    # Contiguous-chunk DistributedSampler: rank r owns indices
    # [r*per_replicas : (r+1)*per_replicas]; map step -> global dataset index.
    per_replicas = math.ceil(len(dataset) / world_size)
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        print(f"[mismatch] mode={mismatch_mode} | "
              f"noise={noise_timesteps} denoise={denoise_timesteps} steps={num_inference_steps}",
              flush=True)
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for step, batch in enumerate(dataloader):

        # ----- resolve image source and condition source per mode -----
        # anchor (= batch) always supplies GT / decode coordinate frame.
        if mismatch_mode == 'matched' or donor_map is None:
            img_src = batch
            cond_src = batch
        else:
            anchor_idx = rank * per_replicas + step  # global index of this anchor sample
            donor_idx = donor_map[anchor_idx % len(dataset)]
            donor = mmcv_collate([dataset[donor_idx]], samples_per_gpu=samples_per_gpu)
            if mismatch_mode == 'cond_gt':
                img_src = donor    # image B (donor)
                cond_src = batch   # condition A == GT (anchor)
            elif mismatch_mode == 'img_gt':
                img_src = batch    # image B == GT (anchor)
                cond_src = donor   # condition A (donor)

        # ----- image-derived BEV feature (single-frame) -----
        latents = bev_model(return_loss=False, only_bev=True,
                            img=img_src['img'], img_metas=img_src['img_metas']).detach()
        latents = latents.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        latents = latents.permute(0, 3, 1, 2)

        # img_metas for the UNet / SegBEVAligner come from the CONDITION source.
        img_metas = cond_src['img_metas'][0].data[0]

        depth_maps = None
        if 'depth_maps' in cond_src.keys():
            depth_maps = torch.stack(cond_src['depth_maps'].data[0], dim=0).to(latents.device)

        # ----- add noise -----
        n_steps = noise_timesteps
        if n_steps > 0:
            if n_steps > 1000:
                latents = torch.randn_like(latents)
                latents = latents * noise_scheduler.init_noise_sigma
            else:
                noise = torch.randn_like(latents)
                n_steps_t = torch.tensor(n_steps).long()
                latents = noise_scheduler.add_noise(latents, noise, n_steps_t)

        # ----- condition-guided denoising -----
        if denoise_timesteps > 0:
            cond = get_condition(cond_src, latents.device, use_cond=True)
            uncond = get_condition(cond_src, latents.device, use_cond=False)
            seg_cond = torch.stack(cond_src['seg_maps'].data[0], dim=0).to(latents.device)
            seg_uncond = get_segmaps_uncond(seg_cond)

            noise_scheduler.config.num_train_timesteps = denoise_timesteps
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

            for _, t in enumerate(noise_scheduler.timesteps):
                t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)
                noise_pred_uncond = unet(latents, t_batch, img_metas, seg_uncond, depth_maps=depth_maps, **uncond)[0]
                noise_pred_cond = unet(latents, t_batch, img_metas, seg_cond, depth_maps=depth_maps, **cond)[0]
                noise_pred = noise_pred_uncond + 2 * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]

        # ----- detection on the (mismatched) BEV, evaluated against ANCHOR GT -----
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

    bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir=os.path.join(save_path, '.dist_test'))
    if have_mask:
        mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir=os.path.join(save_path, '.dist_test'))
    else:
        mask_results = None

    det_results = bbox_results if mask_results is None else {'bbox_results': bbox_results, 'mask_results': mask_results}

    key_score = {}
    if rank == 0:
        eval_kwargs = bev_cfg.get('evaluation', {}).copy()
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs['jsonfile_prefix'] = os.path.join(save_path, det_res_path)
        eval_results = dataset.evaluate(det_results, **eval_kwargs)
        for metric, score in eval_results.items():
            if 'mAP' in metric or 'NDS' in metric:
                key_score[metric] = score
        print(f"[mismatch:{mismatch_mode}] {key_score}", flush=True)
    return key_score


if __name__ == "__main__":
    test()
