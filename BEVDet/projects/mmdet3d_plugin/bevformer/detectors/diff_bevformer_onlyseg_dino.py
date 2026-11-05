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

from projects.bevdiffuser.dinov3_bev_extractor import DINOv3BEVExtractor
from projects.bevdiffuser.layout_diffusion.temporal_dino_condition import (
    TemporalBEVWarper,
    build_temporal_dino_condition,
)

from .bevformer import BEVFormer


class BEVDistillProjector(nn.Module):
    def __init__(self, in_ch=256, hidden_ch=384, out_ch=384, groups=32):
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
class DiffBEVFormerOnlySegDINO(BEVFormer):

    def __init__(self, *args,
                 dino_bev_extractor=None,
                 aux_seg_num_classes=16,
                 aux_seg_weight=0.5,
                 use_proj=False,
                 use_mgd=False,
                 mgd_alpha=0.00002,
                 mgd_lambda=0.65,
                 use_observed_mask_mgd=False,
                 mgd_unobserved_weight=0.0,
                 use_temporal_condition=False,
                 teacher_channels=384,
                 distill_loss_weight=100.0,
                 loss_mix_weight=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if dino_bev_extractor is None:
            raise ValueError('dino_bev_extractor config is required')
        if use_mgd and use_proj:
            raise ValueError('use_mgd and use_proj cannot both be enabled')

        # Lazy, unregistered DINO avoids duplication in model_target and keeps
        # its frozen weights out of the optimizer, DDP and Stage 2 checkpoints.
        object.__setattr__(
            self, '_dino_extractor_cfg', copy.deepcopy(dino_bev_extractor))
        object.__setattr__(self, '_dino_extractor', None)

        self.aux_seg_num_classes = aux_seg_num_classes
        self.aux_seg_weight = aux_seg_weight
        self.use_proj = use_proj
        self.use_mgd = use_mgd
        self.mgd_alpha = mgd_alpha
        self.mgd_lambda = mgd_lambda
        self.use_observed_mask_mgd = use_observed_mask_mgd
        self.mgd_unobserved_weight = mgd_unobserved_weight
        self.use_temporal_condition = use_temporal_condition
        self.teacher_channels = teacher_channels
        self.temporal_warper = None
        if self.use_temporal_condition:
            projector = dino_bev_extractor['projector']
            self.temporal_warper = TemporalBEVWarper(
                projector['bev_h'], projector['bev_w'], projector['pc_range'])
        self.distill_loss_weight = distill_loss_weight
        self.loss_mix_weight = loss_mix_weight
        embed_dim = self.pts_bbox_head.embed_dims

        if use_proj:
            self.bev_distill_proj = BEVDistillProjector(
                embed_dim, self.teacher_channels, self.teacher_channels)

        if use_mgd:
            self.mgd_align = nn.Conv2d(embed_dim, self.teacher_channels, kernel_size=1)
            self.mgd_generation = nn.Sequential(
                nn.Conv2d(self.teacher_channels, self.teacher_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.teacher_channels, self.teacher_channels,
                          kernel_size=3, padding=1),
            )
        elif not use_proj:
            self.bev_channel_align = (
                nn.Identity() if embed_dim == self.teacher_channels
                else nn.Conv2d(embed_dim, self.teacher_channels, kernel_size=1))

    def train_step(self, data, optimizer, model_target=None,
                   bev_diffuser=None, progress=None):
        losses = self(
            **data, model_target=model_target, bev_diffuser=bev_diffuser)
        loss, log_vars = self._parse_losses_mix(
            losses, weight=self.loss_mix_weight)
        return dict(
            loss=loss, log_vars=log_vars,
            num_samples=len(data['img_metas']))

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
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
        len_queue = img.size(1)
        temporal_img = img
        temporal_img_metas = img_metas
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        img_metas = [each[len_queue - 1] for each in img_metas]

        teacher_bev = None
        teacher_observed_mask = None
        if bev_diffuser is not None:
            depth_maps = kwargs.get('depth_maps')
            if isinstance(depth_maps, (list, tuple)):
                depth_maps = torch.stack(depth_maps, dim=0)
            if depth_maps is None:
                raise ValueError(
                    'Metric3D depth_maps are required for DINO BEV projection')

            segmaps = None
            semantic_prob = None
            if 'semantic_bev_probabilities' in kwargs:
                semantic_prob = kwargs['semantic_bev_probabilities']
                if isinstance(semantic_prob, (list, tuple)):
                    semantic_prob = torch.stack(semantic_prob, dim=0)
            else:
                segmaps = kwargs['seg_maps']
                if isinstance(segmaps, (list, tuple)):
                    segmaps = torch.stack(segmaps, dim=0)

            dino_extractor = self.__dict__.get('_dino_extractor')
            if dino_extractor is None:
                dino_cfg = self._dino_extractor_cfg
                dino_extractor = dino_cfg if isinstance(dino_cfg, nn.Module) else DINOv3BEVExtractor(**dino_cfg)
                dino_extractor.requires_grad_(False).eval().to(img.device)
                if dino_extractor.encoder.hidden_size != self.teacher_channels:
                    raise ValueError(
                        f'DINO channels {dino_extractor.encoder.hidden_size} '
                        f'do not match teacher channels {self.teacher_channels}')
                object.__setattr__(
                    self, '_dino_extractor', dino_extractor)

            history_bevs = None
            history_masks = None
            if self.use_temporal_condition:
                temporal_depth = kwargs['temporal_depth_maps']
                temporal_valid = kwargs['temporal_valid_mask']
                if isinstance(temporal_depth, (list, tuple)):
                    temporal_depth = torch.stack(temporal_depth, dim=0)
                if isinstance(temporal_valid, (list, tuple)):
                    temporal_valid = torch.stack(temporal_valid, dim=0)
                dino_result, img_metas, history_bevs, history_masks = (
                    build_temporal_dino_condition(
                        dino_extractor, self.temporal_warper, temporal_img,
                        temporal_depth, temporal_img_metas, temporal_valid))
            else:
                dino_result = dino_extractor(img, img_metas, depth_maps)
            teacher_observed_mask = dino_result['observed_mask'].detach()
            teacher_bev = bev_diffuser(
                dino_result, img_metas, segmaps=segmaps,
                depth_maps=depth_maps, semantic_prob=semantic_prob,
                temporal_bevs=history_bevs, temporal_masks=history_masks)

        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas,
            gt_bboxes_ignore, prev_bev, given_bev,
            model_target=model_target, teacher_bev=teacher_bev,
            teacher_observed_mask=teacher_observed_mask, **kwargs)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          given_bev=None,
                          model_target=None,
                          teacher_bev=None,
                          teacher_observed_mask=None,
                          **kwargs):
        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True)

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev, given_bev=bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        if teacher_bev is not None:
            B = bev.shape[0]
            C = bev.shape[-1]
            H = self.pts_bbox_head.bev_h
            W = self.pts_bbox_head.bev_w
            student_bev = bev.permute(0, 2, 1).reshape(B, C, H, W)
            teacher_bev = teacher_bev.detach().float()

            if self.use_mgd:
                preds_S = self.mgd_align(student_bev)
                mat = torch.rand((B, 1, H, W), device=preds_S.device)
                mat = torch.where(mat > 1 - self.mgd_lambda, torch.zeros_like(mat), torch.ones_like(mat))
                new_bev = self.mgd_generation(preds_S * mat)

                if self.use_observed_mask_mgd:
                    error = (new_bev.float() - teacher_bev).pow(2).mean(dim=1, keepdim=True)
                    observed = teacher_observed_mask.detach().to(device=error.device, dtype=error.dtype)
                    unobserved = 1.0 - observed
                    observed_mse = (error * observed).sum() / (
                        observed.sum().clamp_min(1.0))
                    unobserved_mse = (error * unobserved).sum() / (
                        unobserved.sum().clamp_min(1.0))
                    loss_bev = observed_mse + self.mgd_unobserved_weight * unobserved_mse
                    losses['mgd_observed_mse'] = observed_mse.detach()
                    losses['mgd_unobserved_mse'] = unobserved_mse.detach()
                    losses['teacher_observed_fraction'] = observed.mean().detach()
                else:
                    loss_bev = F.mse_loss(
                        new_bev.float(), teacher_bev, reduction='mean')
                losses['loss_bev'] = loss_bev * self.mgd_alpha

            elif self.use_proj:
                student_bev = self.bev_distill_proj(student_bev.float())
                loss_bev = F.mse_loss(student_bev, teacher_bev, reduction='mean')
                losses['loss_bev'] = (loss_bev * self.distill_loss_weight)

            else:
                student_bev = self.bev_channel_align(student_bev.float())
                loss_bev = F.mse_loss(student_bev, teacher_bev, reduction='mean')
                losses['loss_bev'] = (loss_bev * self.distill_loss_weight)

        return losses

    def _parse_losses_mix(self, losses, weight=0.5):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(
                    _loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(
            value for name, value in log_vars.items() if 'loss' in name)
        bev_loss = sum(
            value for name, value in log_vars.items()
            if 'loss' in name and 'bev' in name)
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
