# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import UNet2DConditionModel
from projects.bevdiffuser.model_utils import build_unet, instantiate_from_config
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler


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
                    
        
        self.unet = instantiate_from_config(unet_cfg)
        assert unet_checkpoint_dir is not None
        self.unet.from_pretrained(unet_checkpoint_dir, subfolder="unet")
        self.unet.requires_grad_(False)
        
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
        
        
    def forward(self, x, condition=None, grad_fn=None): 
        
        cond = condition
        uncond = self.get_dino_uncond(condition)
        
        if self.noise_timesteps > 0:
            noise = torch.randn_like(x)
            noise_timesteps = torch.tensor(self.noise_timesteps).long()
            x = self.noise_scheduler.add_noise(x, noise, noise_timesteps)
            
        if self.denoise_timesteps > 0:
            self.noise_scheduler.config.num_train_timesteps=self.denoise_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
            for _, t in enumerate(self.noise_scheduler.timesteps):
                t_batch = torch.tensor([t] * x.shape[0], device=x.device)
                noise_pred_uncond, noise_pred_cond = self.unet(x, t_batch, **uncond)[0], self.unet(x, t_batch, **cond)[0]
                noise_pred = noise_pred_uncond + 2.0 * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = grad_fn(x) if self.use_classifier_guidence and grad_fn else None # self.use_classifier_guidence=False
                x = self.noise_scheduler.step(noise_pred, t, x, return_dict=False, classifier_gradient=classifier_gradient)[0] 
            
            return x
        
        else:
            timestep = torch.tensor([10] * x.shape[0], device=x.device)  
            multi_feat = self.unet(x, timestep, **cond)[1]  
        
            return multi_feat
