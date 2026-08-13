# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import UNet2DConditionModel
from projects.bevdiffuser.model_utils import build_unet_v2
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler
from projects.bevdiffuser.fm_feature import GetDINOV2Feat

class BEVDiffuser(nn.Module):
    def __init__(self,
                 unet_cfg,
                 unet_checkpoint_dir=None,
                 pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
                 prediction_type=None,
                 noise_timesteps=0,
                 denoise_timesteps=0,
                 num_inference_steps=0,
                 use_classifier_guidence=False,               
                 ):
        super(BEVDiffuser, self).__init__()
        self.noise_scheduler = DDIMGuidedScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        if prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=prediction_type)
                    
        self.unet = build_unet_v2(unet_cfg)
        assert unet_checkpoint_dir is not None
        self.unet.from_pretrained(unet_checkpoint_dir, subfolder="unet")
        self.unet.requires_grad_(False)
        
        self.get_dino = GetDINOV2Feat()
        
        self.noise_timesteps = noise_timesteps
        self.denoise_timesteps = denoise_timesteps
        self.num_inference_steps = num_inference_steps
        self.use_classifier_guidence = use_classifier_guidence
        
        self.auto_denoise_timesteps = False
        if self.denoise_timesteps is None:
            self.auto_denoise_timesteps = True
            
    def get_dino_uncond(self, cond):
            uncond = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in cond.items()}
            last_cls_u = torch.zeros_like(cond['last_cls'])  # (B,V,C_in)
            last_tokens_u = torch.zeros_like(cond['last_tokens'])  # (B,V,N,C_in)
            uncond['last_cls'] = last_cls_u
            uncond['last_tokens'] = last_tokens_u
            return uncond
        
    def get_segmaps_uncond(self, seg_cond):
        uncond = torch.zeros_like(seg_cond)
        return uncond

    def forward(self, x, img, img_metas, segmaps=None, depth_maps=None, grad_fn=None):
         
        if self.noise_timesteps > 0:
            noise = torch.randn_like(x)
            noise_timesteps = torch.tensor(self.noise_timesteps).long()
            x = self.noise_scheduler.add_noise(x, noise, noise_timesteps)
            
        if self.denoise_timesteps > 0:
            dino_cond = self.get_dino(img, img_metas)
            dino_uncond = self.get_dino_uncond(dino_cond)
            # dino_cond, dino_uncond = dino_feat, self.get_dino_uncond(dino_feat)
            seg_cond, seg_uncond = segmaps, self.get_segmaps_uncond(segmaps)

            self.noise_scheduler.config.num_train_timesteps=self.denoise_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
         
            for _, t in enumerate(self.noise_scheduler.timesteps):
                t_batch = torch.tensor([t] * x.shape[0], device=x.device)
                noise_pred_uncond = self.unet(x, t_batch, img_metas, dino_uncond, seg_uncond, depth_maps=depth_maps)
                noise_pred_cond = self.unet(x, t_batch, img_metas, dino_cond, seg_cond, depth_maps=depth_maps)
                noise_pred = noise_pred_uncond + 2.0 * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = grad_fn(x) if self.use_classifier_guidence and grad_fn else None # self.use_classifier_guidence=False
                x = self.noise_scheduler.step(noise_pred, t, x, return_dict=False, classifier_gradient=classifier_gradient)[0] 
                
        return x