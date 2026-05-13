# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
import time
import copy
import numpy as np
import mmdet3d
from .bevformer import BEVFormer

from projects.mmdet3d_plugin.models.utils.bricks import run_time


SEG_LABEL_DIC = {
    'sedan': 1, 'highway': 2, 'bus': 3, 'truck': 4, 'terrain': 5,
    'tree': 6, 'sidewalk': 7, 'bicycle': 8, 'bicyclist': 8,
    'barrier': 9, 'barricade': 9, 'person': 10, 'pedestrian': 10,
    'building': 11, 'bridge': 11, 'pole': 11, 'billboard': 11,
    'light': 11, 'ashbin': 11, 'motorcycle': 12, 'motorcyclist': 12,
    'crane': 13, 'trailer': 14, 'cone': 15, 'sky': 16,
}

CLASS_NAME_TO_SEG_ALIASES = {
    'car': ['sedan'],
    'truck': ['truck'],
    'construction_vehicle': ['crane'],
    'bus': ['bus'],
    'trailer': ['trailer'],
    'barrier': ['barrier', 'barricade'],
    'motorcycle': ['motorcycle', 'motorcyclist'],
    'bicycle': ['bicycle', 'bicyclist'],
    'pedestrian': ['pedestrian', 'person'],
    'traffic_cone': ['cone'],
}

DEFAULT_DET_CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]


def _resolve_fg_seg_labels(class_names):
    fg = set()
    for cn in class_names:
        aliases = CLASS_NAME_TO_SEG_ALIASES.get(cn, [cn])
        for a in aliases:
            if a in SEG_LABEL_DIC:
                fg.add(SEG_LABEL_DIC[a])
    return sorted(fg)


class BEVDistillProjector(nn.Module):
    """
    Student-side feature projector used only for distillation.
    """

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


class BEVDistillProjectorV2(nn.Module):
    """
    Stronger student-side projector with 3x3 spatial mixing and a residual path.
    Higher capacity than the 1x1-only V1, intended to bridge a wider feature gap.
    """

    def __init__(self, in_ch=256, hidden_ch=256, out_ch=256, groups=32):
        super().__init__()
        assert in_ch == out_ch, "Residual path requires in_ch == out_ch"
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, hidden_ch)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, hidden_ch)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.act2(self.gn2(self.conv2(h)))
        return x + self.conv3(h)


@DETECTORS.register_module()
class DiffBEVFormerSeg(BEVFormer):

    def __init__(self, *args,
                 use_proj=False,
                 proj_version='v1',
                 use_seg_mask=False,
                 fg_weight_alpha=5.0,
                 det_class_names=None,
                 use_aux_seg=False,
                 aux_seg_num_classes=16,
                 aux_seg_weight=1.0,
                 aux_seg_valid_threshold=0.1,
                 use_bev_rel=False,
                 bev_rel_weight=10.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_proj = use_proj
        self.proj_version = proj_version
        self.use_seg_mask = use_seg_mask
        self.fg_weight_alpha = fg_weight_alpha
        self.use_aux_seg = use_aux_seg
        self.aux_seg_num_classes = aux_seg_num_classes
        self.aux_seg_weight = aux_seg_weight
        self.aux_seg_valid_threshold = aux_seg_valid_threshold
        self.use_bev_rel = use_bev_rel
        self.bev_rel_weight = bev_rel_weight

        embed_dim = self.pts_bbox_head.embed_dims
        
        if use_proj:
            if proj_version == 'v2':
                self.bev_distill_proj = BEVDistillProjectorV2(embed_dim, embed_dim, embed_dim)
            else:
                self.bev_distill_proj = BEVDistillProjector(embed_dim, embed_dim, embed_dim)
                
        if use_seg_mask:
            cls_names = det_class_names if det_class_names is not None else DEFAULT_DET_CLASS_NAMES
            fg_labels = _resolve_fg_seg_labels(cls_names)
            # NOTE: SegEmbedEncoder clamps labels to [0, 15], so raw label 16
            assert max(fg_labels) <= 15, \
                "FG labels must be in [0,15] due to SegEmbedEncoder.clamp(max=15)."
            self.register_buffer(
                '_fg_seg_idx', torch.tensor(fg_labels, dtype=torch.long), persistent=False,
            )
        if use_aux_seg:
            # Lightweight head: 3x3 Conv-GN-GELU + 1x1 Conv → (num_cls+1) channels.
            self.aux_seg_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, aux_seg_num_classes + 1, kernel_size=1, bias=True),
            )

    def train_step(self, data, optimizer, model_target=None, bev_diffuser=None, progress=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, model_target=model_target, bev_diffuser=bev_diffuser)
        
        weight = 0.5
        # if progress is not None:
        #     weight = max(1 - progress*2, 0.1)
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
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bev_target = None
        if bev_diffuser:
            assert model_target is not None 
            bev_target = model_target(return_loss=False, only_bev=True, img=img, img_metas=img_metas).detach()
            
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, given_bev, model_target, bev_diffuser, bev_target, **kwargs)

        losses.update(losses_pts)
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
                          bev_diffuser=None,
                          bev_target=None,
                          **kwargs):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        # original BEV feature
        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True
        )
        
        losses = dict()
        # task loss
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, given_bev=bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        if bev_diffuser:
            assert bev_target is not None   
              
            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                x_in = x_in.permute(0, 2, 3, 1).reshape(-1, self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w, bev.shape[-1])
                outs = model_target.pts_bbox_head(pts_feats, img_metas, prev_bev=prev_bev, given_bev=x_in)
                losses = model_target.pts_bbox_head.loss(
                    gt_bboxes_list=gt_bboxes_3d,
                    gt_labels_list=gt_labels_3d,
                    preds_dicts=outs,
                    img_metas=img_metas
                )
                loss, _ = self._parse_losses(losses)
                gradient = torch.autograd.grad(loss, x_in)[0]
                gradient = gradient.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
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
            if 'depth_maps' in kwargs.keys():
                depth_maps = torch.stack(kwargs['depth_maps'], dim=0)
                
            bev_ = bev_target.detach() # deno
            bev_ = bev_.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
            bev_, seg_bev_prob = bev_diffuser(bev_, img_metas, get_condition(), segmaps, depth_maps, grad_fn=get_classifier_gradient)
            seg_bev_prob = seg_bev_prob.detach().float()

            B, C = bev.shape[0], bev.shape[-1]
            H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
            bev_ = bev_.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # Student-side projector (optional). Conv requires (B, C, H, W); reshape around it.
            if self.use_proj:
                bev_s = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous().float()
                bev_s = self.bev_distill_proj(bev_s)
                bev_s = bev_s.permute(0, 2, 3, 1).reshape(B, H * W, C)
            else:
                bev_s = bev.float()

            if self.use_seg_mask:
                # FG-weighted MSE: emphasize detection foreground cells 
                det_prob = seg_bev_prob.index_select(1, self._fg_seg_idx).sum(dim=1).clamp(0.0, 1.0)  # (B, H, W)
                weight = (1.0 + self.fg_weight_alpha * det_prob).reshape(B, H * W)                    # (B, H*W)
                mse_loss = ((bev_s - bev_.detach().float()) ** 2).mean(dim=-1)                        # (B, H*W)
                loss_bev = (mse_loss * weight).sum() / weight.sum().clamp_min(1e-6)
            else:
                loss_bev = F.mse_loss(bev_s, bev_.detach().float(), reduction="mean")

            losses['loss_bev'] = loss_bev * 100

            # Relational / structural distillation: match per-cell self-similarity structure.
            if self.use_bev_rel:
                bev_s_raw_2d = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                bev_t_raw_2d = bev_.detach().reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

                sim_s = self._sim_matrix(bev_s_raw_2d.float())
                sim_t = self._sim_matrix(bev_t_raw_2d.float())
                loss_bev_rel = F.mse_loss(sim_s, sim_t.detach())
                losses['loss_bev_rel'] = loss_bev_rel * self.bev_rel_weight

            # Auxiliary seg supervision
            if self.use_aux_seg:
                B = bev.shape[0]
                C = bev.shape[-1]
                H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
                bev_s_raw_2d = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                seg_logits = self.aux_seg_head(bev_s_raw_2d.float())  # (B, num_cls+1, H, W)

                valid = (seg_bev_prob.sum(dim=1) > self.aux_seg_valid_threshold).float()  # (B, H, W)
                log_pred = F.log_softmax(seg_logits, dim=1)
                kl_per_pixel = F.kl_div(log_pred, seg_bev_prob, reduction='none').sum(dim=1)  # (B, H, W)
                loss_aux_seg = (kl_per_pixel * valid).sum() / valid.sum().clamp_min(1.0)
                losses['loss_bev_aux_seg'] = loss_aux_seg * self.aux_seg_weight

        return losses

    @staticmethod
    def _sim_matrix(x):
        """Per-cell cosine self-similarity matrix.

        Args:
            x: (B, C, H, W) feature map.
        Returns:
            S: (B, H*W, H*W) cosine self-similarity matrix.
        """
        x = x.flatten(2)                  # (B, C, H*W)
        x = F.normalize(x, dim=1)         # L2-normalize along channel
        return x.transpose(1, 2) @ x      # (B, H*W, H*W)

    def _parse_losses_mix(self, losses, weight=0.5):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        bev_loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key and 'bev' in _key)
        task_loss = total_loss - bev_loss
        
        loss = (1-weight) * task_loss + weight * bev_loss

        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss
        # log_vars['gq_loss'] = gq_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    
    
    
    
    
@DETECTORS.register_module()
class DiffBEVFormerSegV2(BEVFormer):

    def __init__(self, *args,
                 use_aux_seg=False,
                 aux_seg_num_classes=16,
                 aux_seg_weight=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_aux_seg = use_aux_seg
        self.aux_seg_num_classes = aux_seg_num_classes
        self.aux_seg_weight = aux_seg_weight
        if use_aux_seg:
            embed_dim = self.pts_bbox_head.embed_dims
            # Lightweight head: 3x3 Conv-GN-GELU + 1x1 Conv → (num_cls+1) channels.
            # Predicts BEV-space class probability for KL alignment with seg_aligner.compute_prob_only().
            self.aux_seg_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, aux_seg_num_classes + 1, kernel_size=1, bias=True),
            )

    def train_step(self, data, optimizer, model_target=None, bev_diffuser=None, progress=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, model_target=model_target, bev_diffuser=bev_diffuser)
        
        weight = 0.5
        # if progress is not None:
        #     weight = max(1 - progress*2, 0.1)
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
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bev_target = None
        if bev_diffuser:
            assert model_target is not None 
            bev_target = model_target(return_loss=False, only_bev=True, img=img, img_metas=img_metas).detach()
            
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, given_bev, model_target, bev_diffuser, bev_target, **kwargs)

        losses.update(losses_pts)
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
                          bev_diffuser=None,
                          bev_target=None,
                          **kwargs):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        # original BEV feature
        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True
        )
        
        losses = dict()
        # task loss
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, given_bev=bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        if bev_diffuser:
            assert bev_target is not None   
              
            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                x_in = x_in.permute(0, 2, 3, 1).reshape(-1, self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w, bev.shape[-1])
                outs = model_target.pts_bbox_head(pts_feats, img_metas, prev_bev=prev_bev, given_bev=x_in)
                losses = model_target.pts_bbox_head.loss(
                    gt_bboxes_list=gt_bboxes_3d,
                    gt_labels_list=gt_labels_3d,
                    preds_dicts=outs,
                    img_metas=img_metas
                )
                loss, _ = self._parse_losses(losses)
                gradient = torch.autograd.grad(loss, x_in)[0]
                gradient = gradient.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
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
            if 'depth_maps' in kwargs.keys():
                depth_maps = torch.stack(kwargs['depth_maps'], dim=0)

            bev_ = bev_target.detach() # deno
            bev_ = bev_.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
            bev_ = bev_diffuser(bev_, img_metas, get_condition(), segmaps, depth_maps, grad_fn=get_classifier_gradient)
            bev_flat = bev_.permute(0, 2, 3, 1).reshape(-1, self.pts_bbox_head.bev_h*self.pts_bbox_head.bev_w, bev.shape[-1])
            loss_bev = F.mse_loss(bev.float(), bev_flat.detach().float(), reduction="mean")

            losses['loss_bev'] = loss_bev * 100

            # Auxiliary seg supervision: bridge information asymmetry by forcing
            if self.use_aux_seg:
                B = bev.shape[0]
                C = bev.shape[-1]
                H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
                bev_s_2d = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                seg_logits = self.aux_seg_head(bev_s_2d.float())  # (B, num_cls+1, H, W)

                with torch.no_grad():
                    seg_bev_prob_gt = bev_diffuser.unet.seg_aligner.compute_prob_only(
                                                        segmaps, img_metas, depth_maps,
                                                    ).detach().float()  # (B, num_cls+1, H, W)
                log_pred = F.log_softmax(seg_logits, dim=1)
                # Per-pixel KL(teacher || student) averaged over (B, H, W).
                loss_aux_seg = F.kl_div(log_pred, seg_bev_prob_gt, reduction='none').sum(dim=1).mean()
                losses['loss_bev_aux_seg'] = loss_aux_seg * self.aux_seg_weight

        return losses
    
    def _parse_losses_mix(self, losses, weight=0.5):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        bev_loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key and 'bev' in _key)
        task_loss = total_loss - bev_loss
        
        loss = (1-weight) * task_loss + weight * bev_loss

        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss
        # log_vars['gq_loss'] = gq_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars