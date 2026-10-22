# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import torch
from safetensors.torch import load_file
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
import torch.nn as nn
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import UNet2DConditionModel
from projects.bevdiffuser.model_utils import build_unet_v2
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler


def resolve_unet_checkpoint(path):
    """Accept a Stage 1 checkpoint directory, its unet/ directory or weight file."""
    if path is None:
        raise ValueError('Stage 2 requires a pretrained UNet checkpoint')
    path = os.fspath(path)
    if os.path.isfile(path):
        return path
    for candidate in (os.path.join(path, SAFETENSORS_WEIGHTS_NAME),
                      os.path.join(path, 'unet', SAFETENSORS_WEIGHTS_NAME)):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'No {SAFETENSORS_WEIGHTS_NAME} found in {path} or its unet/ directory')


class BEVDiffuser(nn.Module):
    def __init__(self,
                 unet_cfg,
                 unet_checkpoint_dir=None,
                 pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
                 prediction_type=None,
                 noise_timesteps=0,
                 denoise_timesteps=0,
                 num_inference_steps=0,
                 cfg_scale=2,
                 use_classifier_guidence=False,
                 ):
        super(BEVDiffuser, self).__init__()
        self.noise_scheduler = DDIMGuidedScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        if prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=prediction_type)
                    
        self.unet = build_unet_v2(unet_cfg)
        checkpoint = resolve_unet_checkpoint(unet_checkpoint_dir)
        # Stage 2 freezes this model: missing gates/global embeddings must never
        # silently remain at random initialization through a partial load.
        self.unet.load_state_dict(load_file(checkpoint, device='cpu'), strict=True)
        self.unet.requires_grad_(False)
        self.unet.eval()
        
        self.noise_timesteps = noise_timesteps
        self.denoise_timesteps = denoise_timesteps
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.use_classifier_guidence = use_classifier_guidence
        
        self.auto_denoise_timesteps = False
        if self.denoise_timesteps is None:
            self.auto_denoise_timesteps = True
            
    def get_uncondition(self, cond):
        uncond = {}
        if 'obj_class' in self.unet.layout_encoder.used_condition_types:
            uncond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(self.unet.layout_encoder.num_classes_for_layout_object - 1)
            uncond['obj_class'][:, 0] = self.unet.layout_encoder.num_classes_for_layout_object - 2
        if 'obj_name' in self.unet.layout_encoder.used_condition_types:
            uncond['obj_name'] = cond['default_obj_names']
        if 'obj_bbox' in self.unet.layout_encoder.used_condition_types:
            uncond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
            if self.unet.layout_encoder.use_3d_bbox:
                uncond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
            else:
                uncond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 1, 1])
        uncond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
        uncond['is_valid_obj'][:, 0] = 1.0 
        return uncond
        
    def get_segmaps_uncond(self, seg_cond):
        uncond = torch.zeros_like(seg_cond)
        return uncond

    def forward(self, x, img_metas, segmaps=None, depth_maps=None, grad_fn=None, semantic_prob=None):
        if self.noise_timesteps > 0:
            noise = torch.randn_like(x)
            noise_timesteps = torch.tensor(self.noise_timesteps).long()
            x = self.noise_scheduler.add_noise(x, noise, noise_timesteps)
            
        if self.denoise_timesteps > 0:
            # Cached semantic BEV probabilities replace the live SAM3/Metric3D
            if semantic_prob is not None:
                seg_cond = seg_uncond = None
                cond_kwargs = dict(semantic_prob=semantic_prob)
                uncond_kwargs = dict( semantic_prob=self.get_segmaps_uncond(semantic_prob))
            else:
                seg_cond, seg_uncond = segmaps, self.get_segmaps_uncond(segmaps)
                cond_kwargs, uncond_kwargs = {}, {}

            # seg conditioning does not depend on t: project once, reuse every step
            seg_bev_cond = self.unet.encode_seg(seg_cond, img_metas, depth_maps=depth_maps, **cond_kwargs)
            seg_bev_uncond = self.unet.encode_seg(seg_uncond, img_metas, depth_maps=depth_maps, **uncond_kwargs)

            self.noise_scheduler.config.num_train_timesteps=self.denoise_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

            for t in self.noise_scheduler.timesteps:
                t_batch = torch.tensor([t] * x.shape[0], device=x.device)
                noise_pred_uncond = self.unet(x, t_batch, img_metas, None, depth_maps=depth_maps, seg_bev_maps=seg_bev_uncond)
                noise_pred_cond = self.unet(x, t_batch, img_metas, None, depth_maps=depth_maps, seg_bev_maps=seg_bev_cond)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = grad_fn(x) if self.use_classifier_guidence and grad_fn else None # self.use_classifier_guidence=False
                x = self.noise_scheduler.step(noise_pred, t, x, return_dict=False, classifier_gradient=classifier_gradient)[0] 
            
        return x
