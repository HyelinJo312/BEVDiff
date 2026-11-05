# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import copy
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

from .bevformer_occ import BEVFormerOcc


@DETECTORS.register_module()
class DiffBEVFormerOcc(BEVFormerOcc):
    """Stage-2 BEV distillation model for native BEVFormer Occ3D.

    The image encoder, temporal BEV encoder, occupancy decoder/predictor and
    task loss are inherited unchanged from :class:`BEVFormerOcc`.  A frozen
    Stage-1 diffuser supplies the teacher BEV used by the optional MGD loss.
    """

    def __init__(self, *args, use_mgd=False, mgd_alpha=0.00002,
                 mgd_lambda=0.65, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= mgd_lambda <= 1.0:
            raise ValueError('mgd_lambda must be in [0, 1]')
        if mgd_alpha < 0:
            raise ValueError('mgd_alpha must be non-negative')
        self.use_mgd = use_mgd
        self.mgd_alpha = mgd_alpha
        self.mgd_lambda = mgd_lambda
        self.loss_mix_weight = 0.5
        self._gradient_monitor_context = None
        self._gradient_monitor_pending = None

        if use_mgd:
            embed_dim = self.pts_bbox_head.embed_dims
            # Occupancy Stage 1/2 use the same 256-channel BEV space, so no
            # separate projection is required.
            self.mgd_align = None
            self.mgd_generation = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Preserve the caller's mode around BEVFormerOcc's history pass."""
        was_training = self.training
        bev = super().obtain_history_bev(imgs_queue, img_metas_list)
        self.train(was_training)
        return bev

    def forward(self, return_loss=True, only_bev=False, **kwargs):
        # model_target uses this path to obtain the clean epoch-24 BEV target.
        if only_bev:
            if return_loss:
                raise ValueError('only_bev is an inference-only interface')
            img = kwargs.get('img')
            if not isinstance(img, torch.Tensor):
                raise TypeError('Stage-2 teacher extraction expects a temporal image tensor')
            return self.obtain_history_bev(img, copy.deepcopy(kwargs['img_metas']))
        return super().forward(return_loss=return_loss, **kwargs)

    def train_step(self, data, optimizer, model_target=None,
                   bev_diffuser=None, progress=None):
        try:
            losses = self(**data, model_target=model_target,
                          bev_diffuser=bev_diffuser)
            loss, log_vars = self._parse_losses_mix(losses)
            return dict(loss=loss, log_vars=log_vars,
                        num_samples=len(data['img_metas']))
        finally:
            self._gradient_monitor_context = None

    @staticmethod
    def _stack_condition(value, device):
        if isinstance(value, (list, tuple)):
            value = torch.stack(value, dim=0)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Diffusion condition must be a tensor, got {type(value)}')
        return value.to(device=device, non_blocking=True)

    def _bev_to_map(self, bev):
        """Accept native Occ token/map layouts and return [B,C,Y,X]."""
        height = self.pts_bbox_head.bev_h
        width = self.pts_bbox_head.bev_w
        channels = self.pts_bbox_head.embed_dims
        if bev.ndim == 4 and tuple(bev.shape[1:]) == (channels, height, width):
            return bev.contiguous()
        if bev.ndim == 3 and tuple(bev.shape[1:]) == (height * width, channels):
            return bev.transpose(1, 2).reshape(-1, channels, height, width).contiguous()
        if (bev.ndim == 3 and bev.shape[0] == height * width
                and bev.shape[2] == channels):
            return bev.permute(1, 2, 0).reshape(-1, channels, height, width).contiguous()
        raise ValueError(f'Unexpected occupancy BEV shape: {tuple(bev.shape)}')

    @staticmethod
    def _decode_occ_map(model, bev_map):
        """Run the native TransformerOcc post-BEV decoder without an encoder."""
        core = getattr(model, 'module', model)
        transformer = core.pts_bbox_head.transformer
        if not transformer.use_3d:
            raise ValueError('DiffBEVFormerOcc requires TransformerOcc(use_3d=True)')
        batch, _, height, width = bev_map.shape
        volume = bev_map.reshape(batch, -1, transformer.pillar_h, height, width)
        features = transformer.decoder(volume).permute(0, 4, 3, 2, 1)
        return dict(bev_embed=bev_map, occ=transformer.predicter(features))

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                      gt_labels_3d=None, voxel_semantics=None,
                      mask_lidar=None, mask_camera=None, gt_labels=None,
                      gt_bboxes=None, img=None, proposals=None,
                      gt_bboxes_ignore=None, img_depth=None, img_mask=None,
                      given_bev=None, model_target=None, bev_diffuser=None,
                      **kwargs):
        if voxel_semantics is None:
            raise ValueError('DiffBEVFormerOcc requires voxel_semantics')

        bev_target = None
        if bev_diffuser is not None:
            if model_target is None:
                raise ValueError('A frozen model_target is required with bev_diffuser')
            with torch.no_grad():
                bev_target = model_target(
                    return_loss=False, only_bev=True, img=img,
                    img_metas=copy.deepcopy(img_metas)).detach()

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        current_img = img[:, -1, ...]
        prev_bev = self.obtain_history_bev(prev_img, copy.deepcopy(img_metas))

        current_metas = [each[len_queue - 1] for each in img_metas]
        if not current_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=current_img, img_metas=current_metas)
        return self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, voxel_semantics,
            mask_camera, current_metas, gt_bboxes_ignore, prev_bev,
            model_target=model_target, bev_diffuser=bev_diffuser,
            bev_target=bev_target, **kwargs)

    def forward_pts_train(self, 
                          pts_feats, 
                          gt_bboxes_3d, 
                          gt_labels_3d,
                          voxel_semantics, 
                          mask_camera, 
                          img_metas,
                          gt_bboxes_ignore=None, 
                          prev_bev=None,
                          model_target=None, 
                          bev_diffuser=None,
                          bev_target=None, **kwargs):

        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True
        )
        
        losses = dict()
        # task loss
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, given_bev=bev)
        losses = self.pts_bbox_head.loss(
            voxel_semantics, mask_camera, outs, img_metas=img_metas)

        if bev_diffuser:
            assert bev_target is not None   
            
            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                target_core = getattr(model_target, 'module', model_target)
                task_outs = self._decode_occ_map(target_core, x_in)
                task_losses = target_core.pts_bbox_head.loss(
                    voxel_semantics, mask_camera, task_outs,
                    img_metas=img_metas)
                task_loss, _ = target_core._parse_losses(task_losses)
                return torch.autograd.grad(task_loss, x_in)[0]
            
            noisy_bev = bev_target.detach()
            teacher_input = self._bev_to_map(noisy_bev)
            device = teacher_input.device
            
            segmaps = None
            depth_maps = None
            semantic_prob = None
            if 'semantic_bev_probabilities' in kwargs:
                semantic_prob = self._stack_condition(kwargs['semantic_bev_probabilities'], device)
            else:
                if 'seg_maps' not in kwargs:
                    raise KeyError('Stage 2 requires semantic_bev_probabilities or seg_maps')
                segmaps = self._stack_condition(kwargs['seg_maps'], device)
                if 'depth_maps' in kwargs:
                    depth_maps = self._stack_condition(kwargs['depth_maps'], device)

            teacher_bev = bev_diffuser(teacher_input, img_metas, segmaps, depth_maps, 
                                       grad_fn=get_classifier_gradient, semantic_prob=semantic_prob)
            student_bev = self._bev_to_map(bev)
            
            if teacher_bev.shape != student_bev.shape:
                raise ValueError(
                    f'Teacher/student BEV mismatch: {tuple(teacher_bev.shape)} vs '
                    f'{tuple(student_bev.shape)}')

            if self.use_mgd:
                if self.mgd_align is not None:
                    preds_student = self.mgd_align(student_bev)
                else:
                    preds_student = student_bev
                batch, _, height, width = preds_student.shape
                mat = torch.rand((batch, 1, height, width), device=preds_student.device)
                mat = torch.where(mat > 1 - self.mgd_lambda, torch.zeros_like(mat), torch.ones_like(mat))
                if self._gradient_monitor_context is not None:
                    mask_counts = (mat.detach().double().sum(), mat.numel())
                generated_bev = self.mgd_generation(torch.mul(preds_student, mat))
                loss_bev = F.mse_loss(generated_bev, teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * self.mgd_alpha
            else:
                loss_bev = F.mse_loss(student_bev.float(), teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * 100

            if self._gradient_monitor_context is not None:
                from projects.mmdet3d_plugin.models.hooks.occ_gradient_monitor import gradient_sufficient_stats
                task_terms = []
                for name, value in losses.items():
                    if 'loss' in name and 'bev' not in name:
                        task_terms.append(sum(item.mean() for item in value)
                                          if isinstance(value, list) else value.mean())
                task_loss = sum(task_terms)
                task_gradient = torch.autograd.grad(
                    (1.0 - self.loss_mix_weight) * task_loss, bev,
                    retain_graph=True, create_graph=False, allow_unused=True)[0]
                kd_gradient = torch.autograd.grad(
                    self.loss_mix_weight * losses['loss_bev'].mean(), bev,
                    retain_graph=True, create_graph=False, allow_unused=True)[0]
                self._gradient_monitor_pending = gradient_sufficient_stats(
                    self._bev_to_map(task_gradient) if task_gradient is not None else None,
                    self._bev_to_map(kd_gradient) if kd_gradient is not None else None,
                    self._bev_to_map(bev),
                    mask_counts if self.use_mgd else None)
                del task_gradient, kd_gradient

        return losses

    def _parse_losses_mix(self, losses, weight=None):
        """Match the Stage-2 detector's task/distillation loss mixing."""
        if weight is None:
            weight = self.loss_mix_weight
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(value.mean() for value in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(value for name, value in log_vars.items()
                         if 'loss' in name)
        bev_loss = sum(value for name, value in log_vars.items()
                       if 'loss' in name and 'bev' in name)
        task_loss = total_loss - bev_loss
        loss = (1.0 - weight) * task_loss + weight * bev_loss
        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss

        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars
