# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Semantic/CLS-conditioned Stage 2 diffusion inference for DINOv3 BEV."""

import os

import torch
import torch.nn as nn
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import load_file

from projects.bevdiffuser.model_utils import build_unet_v2
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler


def resolve_unet_checkpoint(path):
    """Resolve a Stage 1 checkpoint root, ``unet/`` directory, or weight file."""
    if path is None:
        raise ValueError('Stage 2 requires a pretrained DINO UNet checkpoint')
    path = os.fspath(path)
    if os.path.isfile(path):
        return path
    candidates = (
        os.path.join(path, SAFETENSORS_WEIGHTS_NAME),
        os.path.join(path, 'unet', SAFETENSORS_WEIGHTS_NAME),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f'No {SAFETENSORS_WEIGHTS_NAME} found in {path} or its unet/ directory')


class BEVDiffuser(nn.Module):
    """Frozen Stage 1 UNet operating on precomputed DINOv3 BEV features."""

    def __init__(self, unet_cfg, unet_checkpoint_dir=None,
                 pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1',
                 prediction_type=None, noise_timesteps=0, denoise_timesteps=0,
                 num_inference_steps=0, use_classifier_guidence=False):
        super().__init__()
        if use_classifier_guidence:
            raise ValueError(
                'Classifier guidance is unavailable for the 384-channel DINO teacher; '
                'the BEVFormer detection head consumes 256 channels.')
        if noise_timesteps < 0 or (denoise_timesteps is not None
                                  and denoise_timesteps < 0):
            raise ValueError(
                'noise_timesteps and denoise_timesteps must be non-negative')
        if denoise_timesteps and num_inference_steps < 1:
            raise ValueError(
                'num_inference_steps must be positive when denoising is enabled')

        self.unet = build_unet_v2(unet_cfg)
        checkpoint = resolve_unet_checkpoint(unet_checkpoint_dir)
        self.unet.load_state_dict(
            load_file(checkpoint, device='cpu'), strict=True)
        self.unet.requires_grad_(False).eval()

        self.noise_scheduler = DDIMGuidedScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder='scheduler')
        if prediction_type is not None:
            self.noise_scheduler.register_to_config(
                prediction_type=prediction_type)

        self.noise_timesteps = int(noise_timesteps)
        self.denoise_timesteps = (None if denoise_timesteps is None
                                  else int(denoise_timesteps))
        self.num_inference_steps = int(num_inference_steps)
        self.use_classifier_guidence = False
        self.auto_denoise_timesteps = self.denoise_timesteps is None

    def train(self, mode=True):
        """Keep the frozen Stage 1 UNet in evaluation mode."""
        super().train(False)
        self.unet.eval()
        return self

    @staticmethod
    def _zero_condition(condition):
        return torch.zeros_like(condition)

    def _encode_semantic_condition(self, img_metas, segmaps, depth_maps,
                                   semantic_prob):
        if semantic_prob is not None:
            if segmaps is not None:
                raise ValueError('Provide semantic_prob or segmaps, not both')
            cond = self.unet.encode_seg(
                None, img_metas, semantic_prob=semantic_prob)
            uncond = self.unet.encode_seg(
                None, img_metas,
                semantic_prob=self._zero_condition(semantic_prob))
            return cond, uncond
        if segmaps is None:
            raise ValueError(
                'segmaps are required when semantic_prob is absent')
        cond = self.unet.encode_seg(
            segmaps, img_metas, depth_maps=depth_maps)
        uncond = self.unet.encode_seg(
            self._zero_condition(segmaps), img_metas,
            depth_maps=depth_maps)
        return cond, uncond

    def _unpack_dino_result(self, dino_result):
        if not isinstance(dino_result, dict):
            raise TypeError('dino_result must be a dictionary')
        if 'bev_features' not in dino_result or 'cls_tokens' not in dino_result:
            raise KeyError(
                "dino_result must contain 'bev_features' and 'cls_tokens'")
        latents = dino_result['bev_features']
        cls_tokens = dino_result['cls_tokens']
        if latents.ndim != 4:
            raise ValueError('DINO BEV features must have shape [B, C, H, W]')
        if cls_tokens.ndim != 3 or cls_tokens.shape[0] != latents.shape[0]:
            raise ValueError('DINO CLS tokens must have shape [B, V, C]')
        if latents.shape[1] != self.unet.in_channels:
            raise ValueError(
                f'DINO BEV has {latents.shape[1]} channels but UNet expects '
                f'{self.unet.in_channels}')
        return latents.detach(), cls_tokens.detach()

    @torch.no_grad()
    def forward(self, dino_result, img_metas, segmaps=None, depth_maps=None,
                semantic_prob=None, temporal_bevs=None, temporal_masks=None):
        """Noise and denoise a detector-provided DINO BEV latent."""
        sample, cls_tokens = self._unpack_dino_result(dino_result)
        temporal_contexts = None
        if self.unet.use_temporal_condition:
            temporal_contexts = self.unet.encode_temporal(
                temporal_bevs, temporal_masks)

        if self.noise_timesteps > 0:
            noise = torch.randn_like(sample)
            timesteps = torch.full((sample.shape[0],), self.noise_timesteps, device=sample.device, dtype=torch.long)
            sample = self.noise_scheduler.add_noise(sample, noise, timesteps)

        if self.denoise_timesteps > 0:
            seg_cond, seg_uncond = self._encode_semantic_condition(img_metas, segmaps, depth_maps, semantic_prob)
            self.noise_scheduler.config.num_train_timesteps = self.denoise_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps, device=sample.device)
            cond_keep = torch.ones(sample.shape[0], device=sample.device, dtype=torch.bool)
            uncond_keep = torch.zeros_like(cond_keep)

            for timestep in self.noise_scheduler.timesteps:
                t_batch = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
                pred_uncond = self.unet(
                    sample, t_batch, img_metas, seg_bev_maps=seg_uncond,
                    cls_tokens=cls_tokens, cls_keep_mask=uncond_keep,
                    temporal_contexts=temporal_contexts,
                    temporal_keep_mask=uncond_keep)
                pred_cond = self.unet(
                    sample, t_batch, img_metas, seg_bev_maps=seg_cond,
                    cls_tokens=cls_tokens, cls_keep_mask=cond_keep,
                    temporal_contexts=temporal_contexts,
                    temporal_keep_mask=cond_keep)
                prediction = pred_uncond + 2 * (pred_cond - pred_uncond)
                sample = self.noise_scheduler.step(prediction, timestep, sample, return_dict=False)[0]

        return sample
