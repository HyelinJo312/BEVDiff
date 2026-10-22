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


class BEVDistillProjector(nn.Module):
    def __init__(self, in_ch=256, hidden_ch=256, out_ch=256, groups=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.proj(x)


@DETECTORS.register_module()
class DiffBEVFormerSeg(BEVFormer):

    def __init__(self, *args,
                 use_proj=False,
                 aux_seg_num_classes=16,
                 aux_seg_weight=1.0,
                 aux_seg_valid_threshold=0.1,
                 use_cos_distill=False,
                 cos_distill_weight=50.0,
                 use_bbox_weight=False,
                 bbox_weight_min_overlap=0.1,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_proj = use_proj
        self.aux_seg_num_classes = aux_seg_num_classes
        self.aux_seg_weight = aux_seg_weight
        self.aux_seg_valid_threshold = aux_seg_valid_threshold
        self.use_cos_distill = use_cos_distill
        self.cos_distill_weight = cos_distill_weight
        self.use_bbox_weight = use_bbox_weight
        self.bbox_weight_min_overlap = bbox_weight_min_overlap
        self.pc_range = tuple(pc_range)

        embed_dim = self.pts_bbox_head.embed_dims
        
        if self.use_proj:
            self.bev_distill_proj = BEVDistillProjector(embed_dim, embed_dim, embed_dim)

    def _compute_fg_heatmap(self, gt_bboxes_3d, bev_h, bev_w, device):
        """Returns (B, bev_h, bev_w) heatmap with Gaussian peaks at object centers."""
        B = len(gt_bboxes_3d)
        fg_map = torch.zeros((B, bev_h, bev_w), device=device, dtype=torch.float32)

        pc_x_min, pc_y_min = self.pc_range[0], self.pc_range[1]
        voxel_x = (self.pc_range[3] - self.pc_range[0]) / bev_w
        voxel_y = (self.pc_range[4] - self.pc_range[1]) / bev_h

        for b, gt_bboxes in enumerate(gt_bboxes_3d):
            if len(gt_bboxes) == 0:
                continue
            centers = gt_bboxes.gravity_center.to(device)
            sizes = gt_bboxes.tensor[:, 3:6].to(device)

            for k in range(len(gt_bboxes)):
                w_cells = sizes[k, 0] / voxel_x
                l_cells = sizes[k, 1] / voxel_y
                if w_cells.item() <= 0 or l_cells.item() <= 0:
                    continue
                radius = gaussian_radius((l_cells, w_cells), min_overlap=self.bbox_weight_min_overlap)
                radius = max(1, int(radius))
                cx_int = int(((centers[k, 0] - pc_x_min) / voxel_x).item())
                cy_int = int(((centers[k, 1] - pc_y_min) / voxel_y).item())
                if 0 <= cx_int < bev_w and 0 <= cy_int < bev_h:
                    draw_heatmap_gaussian(fg_map[b], (cx_int, cy_int), radius)

        return fg_map  # (B, bev_h, bev_w), in [0, 1]

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
            # detection only
            losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs, img_metas=img_metas)
        elif 'all_bbox_preds' not in outs.keys():
            # segmentation only
            losses = self.pts_bbox_head.seg_loss(semantic_indices, outs)
        else:
            # multi-task
            losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs, img_metas=img_metas)
            seg_losses = self.pts_bbox_head.seg_loss(semantic_indices, outs)
            losses.update(seg_losses)

        if bev_diffuser:
            assert bev_target is not None

            B = bev.shape[0]
            C = bev.shape[-1]
            H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w

            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                x_in = x_in.permute(0, 2, 3, 1).reshape(-1, H * W, C)
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
                gradient = gradient.reshape(B, H, W, C).permute(0, 3, 1, 2)
                return gradient

            def get_condition():
                cond = {}
                if 'layout_obj_classes' in kwargs:
                    cond['obj_class'] = torch.stack(kwargs['layout_obj_classes'])
                if 'layout_obj_bboxes' in kwargs:
                    cond['obj_bbox'] = torch.stack(kwargs['layout_obj_bboxes'])
                if 'layout_obj_is_valid' in kwargs:
                    cond['is_valid_obj'] = torch.stack(kwargs['layout_obj_is_valid'])
                if 'layout_obj_names' in kwargs:
                    cond['obj_name'] = torch.stack(kwargs['layout_obj_names'])
                if 'default_obj_names' in kwargs:
                    cond['default_obj_names'] = torch.stack(kwargs['default_obj_names'])
                return cond

            segmaps = torch.stack(kwargs['seg_maps'], dim=0)
            depth_maps = None
            if 'depth_maps' in kwargs:
                depth_maps = torch.stack(kwargs['depth_maps'], dim=0)

            # teacher BEV: noisy → diffuser → denoised
            bev_ = bev_target.detach()
            bev_ = bev_.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            bev_ = bev_diffuser(bev_, img_metas, get_condition(), segmaps, depth_maps,grad_fn=get_classifier_gradient)
            bev_ = bev_.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)

            # student BEV (optionally projected)
            bev_s_2d = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous().float()
            if self.use_proj:
                bev_s_2d = self.bev_distill_proj(bev_s_2d)
            bev_s = bev_s_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)

            loss_bev = F.mse_loss(bev_s, bev_.detach().float(), reduction='mean')

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


@DETECTORS.register_module()
class DiffBEVFormerSegV2(BEVFormer):
    def __init__(self, *args,
                 use_mgd=False,
                 mgd_alpha=100,
                 mgd_lambda=0.6,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mgd = use_mgd
        self.mgd_alpha = mgd_alpha
        self.mgd_lambda = mgd_lambda

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

            def get_condition():
                cond = {}
                if 'layout_obj_classes' in kwargs:
                    cond['obj_class'] = torch.stack(kwargs['layout_obj_classes'])
                if 'layout_obj_bboxes' in kwargs:
                    cond['obj_bbox'] = torch.stack(kwargs['layout_obj_bboxes'])
                if 'layout_obj_is_valid' in kwargs:
                    cond['is_valid_obj'] = torch.stack(kwargs['layout_obj_is_valid'])
                if 'layout_obj_names' in kwargs:
                    cond['obj_name'] = torch.stack(kwargs['layout_obj_names'])
                if 'default_obj_names' in kwargs:
                    cond['default_obj_names'] = torch.stack(kwargs['default_obj_names'])
                return cond

            segmaps = torch.stack(kwargs['seg_maps'], dim=0)
            depth_maps = None
            if 'depth_maps' in kwargs:
                depth_maps = torch.stack(kwargs['depth_maps'], dim=0)

            B = bev.shape[0]
            C = bev.shape[-1]
            H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w

            # teacher BEV: noisy → diffuser → denoised (B, C, H, W)
            noisy_bev = bev_target.detach()
            noisy_bev = noisy_bev.reshape(B, H, W, C).permute(0, 3, 1, 2)
            teacher_bev, seg_bev_prob = bev_diffuser(
                noisy_bev, img_metas, get_condition(), segmaps, depth_maps,
                grad_fn=get_classifier_gradient)
            seg_bev_prob = seg_bev_prob.detach().float()  # (B, num_cls+1, H, W)

            student_bev = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # (B, C, H, W)

            if self.use_mgd:
                preds_S = self.mgd_align(student_bev) if self.mgd_align is not None else student_bev
                mat = torch.rand((B, 1, H, W), device=preds_S.device)
                mat = torch.where(mat > 1 - self.mgd_lambda,
                                  torch.zeros_like(mat), torch.ones_like(mat))
                new_bev = self.mgd_generation(torch.mul(preds_S, mat))
                loss_bev = F.mse_loss(new_bev, teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * self.mgd_alpha

            elif self.use_proj:
                bev_s_2d = self.bev_distill_proj(student_bev.float())
                bev_s_flat = bev_s_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)
                teacher_bev_flat = teacher_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)
                loss_bev = F.mse_loss(
                    bev_s_flat, teacher_bev_flat.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * 100

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


