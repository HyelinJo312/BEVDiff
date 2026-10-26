# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, draw_heatmap_gaussian, gaussian_radius
import copy
import numpy as np
from .bevformer import BEVFormer

from projects.mmdet3d_plugin.models.utils.bricks import run_time


def sample_spatialchannel_keep_mask(x, cell_ratio, channel_ratio):
    """Drop a channel subset independently inside selected BEV cells."""
    if x.ndim != 4:
        raise ValueError('MGD input must have shape [B, C, H, W]')
    for name, ratio in (('cell_ratio', cell_ratio), ('channel_ratio', channel_ratio)):
        if not 0 <= ratio <= 1:
            raise ValueError(f'{name} must be between 0 and 1')
    B, C, H, W = x.shape
    cells = torch.rand((B, 1, H, W), device=x.device) < cell_ratio
    channels = torch.rand((B, C, H, W), device=x.device) < channel_ratio
    return (~(cells & channels)).to(dtype=x.dtype)



@DETECTORS.register_module()
class DiffBEVFormerMapSeg(BEVFormer):
    def __init__(self, *args,
                 use_mgd=False,
                 mgd_alpha=100,
                 mgd_lambda=0.6,
                 use_spatialchannel=False,
                 spatialchannel_cell_ratio=0.6,
                 spatialchannel_channel_ratio=0.85,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mgd = use_mgd
        self.mgd_alpha = mgd_alpha
        self.mgd_lambda = mgd_lambda
        self.use_spatialchannel = use_spatialchannel
        self.spatialchannel_cell_ratio = spatialchannel_cell_ratio
        self.spatialchannel_channel_ratio = spatialchannel_channel_ratio
        if not 0 <= mgd_lambda <= 1:
            raise ValueError('mgd_lambda must be between 0 and 1')
        for name, ratio in (('spatialchannel_cell_ratio', spatialchannel_cell_ratio),
                            ('spatialchannel_channel_ratio', spatialchannel_channel_ratio)):
            if not 0 <= ratio <= 1:
                raise ValueError(f'{name} must be between 0 and 1')

        embed_dim = self.pts_bbox_head.embed_dims

        if use_mgd:
            self.mgd_align = None
            self.mgd_generation = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )

    def train_step(self, data, optimizer, model_target=None, bev_diffuser=None, progress=None):
        losses = self(**data, model_target=model_target, bev_diffuser=bev_diffuser)

        weight = 0.5
        loss, log_vars = self._parse_losses_mix(losses, weight=weight)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      semantic_indices=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      given_bev=None,
                      model_target=None,
                      bev_diffuser=None,
                      **kwargs):
        bev_target = None
        if bev_diffuser:
            assert model_target is not None
            with torch.no_grad():
                bev_target = model_target(
                    return_loss=False, only_bev=True, img=img, img_metas=img_metas
                ).detach()

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, semantic_indices,
            img_metas, gt_bboxes_ignore, prev_bev, given_bev,
            model_target, bev_diffuser, bev_target, **kwargs)

        losses = dict()
        losses.update(losses_pts)
        return losses

    def run_teacher(self, bev_diffuser, noisy_bev, img_metas, grad_fn, **kwargs):
        """Denoise the frozen target BEV with the semantic-prior teacher."""
        segmaps = None
        depth_maps = None
        semantic_prob = None
        if 'semantic_bev_probabilities' in kwargs:
            semantic_prob = torch.stack(kwargs['semantic_bev_probabilities'], dim=0).to(device=noisy_bev.device, dtype=torch.float32)
            expected_shape = (noisy_bev.shape[0], bev_diffuser.unet.seg_aligner.input_channels,
                              self.pts_bbox_head.bev_h,
                              self.pts_bbox_head.bev_w)
        else:
            segmaps = torch.stack(kwargs['seg_maps'], dim=0)
            if 'depth_maps' in kwargs:
                depth_maps = torch.stack(kwargs['depth_maps'], dim=0)
        return bev_diffuser(noisy_bev, img_metas, segmaps, depth_maps, grad_fn=grad_fn, semantic_prob=semantic_prob)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          semantic_indices,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          given_bev=None,
                          model_target=None,
                          bev_diffuser=None,
                          bev_target=None,
                          **kwargs):
        # student BEV feature
        bev = self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=True)

        # task loss (det / seg / multi-task)
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev, given_bev=bev)

        if 'seg_preds' not in outs.keys():
            losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs, img_metas=img_metas)
        elif 'all_bbox_preds' not in outs.keys():
            losses = self.pts_bbox_head.seg_loss(semantic_indices, outs)
        else:
            losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs, img_metas=img_metas)
            seg_losses = self.pts_bbox_head.seg_loss(semantic_indices, outs)
            losses.update(seg_losses)

        if bev_diffuser:
            assert bev_target is not None

            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                x_in = x_in.permute(0, 2, 3, 1).reshape(
                    -1, self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w, bev.shape[-1])
                t_outs = model_target.pts_bbox_head(
                    pts_feats, img_metas, prev_bev=prev_bev, given_bev=x_in)
                if 'seg_preds' not in t_outs.keys():
                    t_losses = model_target.pts_bbox_head.loss(
                        gt_bboxes_list=gt_bboxes_3d,
                        gt_labels_list=gt_labels_3d,
                        preds_dicts=t_outs,
                        img_metas=img_metas)
                elif 'all_bbox_preds' not in t_outs.keys():
                    t_losses = model_target.pts_bbox_head.seg_loss(semantic_indices, t_outs)
                else:
                    t_losses = model_target.pts_bbox_head.loss(
                        gt_bboxes_list=gt_bboxes_3d,
                        gt_labels_list=gt_labels_3d,
                        preds_dicts=t_outs,
                        img_metas=img_metas)
                    seg_t_losses = model_target.pts_bbox_head.seg_loss(semantic_indices, t_outs)
                    t_losses.update(seg_t_losses)
                loss, _ = self._parse_losses(t_losses)
                gradient = torch.autograd.grad(loss, x_in)[0]
                gradient = gradient.reshape(
                    -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w,
                    bev.shape[-1]).permute(0, 3, 1, 2)
                return gradient


            noisy_bev = bev_target.detach()
            noisy_bev = noisy_bev.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
            teacher_bev = self.run_teacher(bev_diffuser, noisy_bev, img_metas,
                                           get_classifier_gradient, **kwargs)
            # seg_bev_prob = seg_bev_prob.detach().float()  # (B, num_cls+1, H, W)

            B = bev.shape[0]
            C = bev.shape[-1]
            H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
            # teacher_bev: (B, C, H, W) — keep 2D for MGD; flatten inline when needed
            student_bev = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # (B, C, H, W)
            if teacher_bev.shape != student_bev.shape:
                raise ValueError(f'Student/teacher BEV shape mismatch: '
                                 f'{tuple(student_bev.shape)} vs {tuple(teacher_bev.shape)}')

            if self.use_mgd:
                if self.mgd_align is not None:
                    preds_S = self.mgd_align(student_bev) 
                else:
                    preds_S = student_bev
                if self.use_spatialchannel:
                    mat = sample_spatialchannel_keep_mask(
                        preds_S, self.spatialchannel_cell_ratio,
                        self.spatialchannel_channel_ratio)
                else:
                    mat = torch.rand((B, 1, H, W), device=preds_S.device)
                    mat = torch.where(mat > 1 - self.mgd_lambda, torch.zeros_like(mat), torch.ones_like(mat))
                new_bev = self.mgd_generation(torch.mul(preds_S, mat))
                loss_bev = F.mse_loss(new_bev, teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * self.mgd_alpha
                
            else:
                teacher_bev = teacher_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)
                loss_bev = F.mse_loss(bev.float(), teacher_bev.detach().float(), reduction="mean")
                losses['loss_bev'] = loss_bev * 10

        return losses


    def _parse_losses_mix(self, losses, weight=0.5):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        bev_loss = sum(_value for _key, _value in log_vars.items()
                       if 'loss' in _key and 'bev' in _key)
        task_loss = total_loss - bev_loss

        loss = (1 - weight) * task_loss + weight * bev_loss

        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


