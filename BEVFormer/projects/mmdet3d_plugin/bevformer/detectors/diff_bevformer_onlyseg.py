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
import time
import copy
import numpy as np
import mmdet3d
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
    
    
    
class MGDAlign(nn.Module):
    def __init__(self, dim, groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(groups, dim, affine=False)
        self.dw_affine = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        # nn.init.dirac_(self.dw_affine.weight)  # identity at init (center=1, else=0)
        # nn.init.zeros_(self.dw_affine.bias)

    def forward(self, x):
        return self.dw_affine(self.gn(x))
    
    
@DETECTORS.register_module()
class DiffBEVFormerSegV3(BEVFormer):

    def __init__(self, *args,
                 use_aux_seg=False,
                 aux_seg_num_classes=16,
                 aux_seg_weight=0.5,
                 use_proj=False,
                 use_mgd=False,
                 mgd_alpha=0.00002,
                 mgd_lambda=0.65,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_aux_seg = use_aux_seg
        self.aux_seg_num_classes = aux_seg_num_classes
        self.aux_seg_weight = aux_seg_weight
        self.use_proj = use_proj
        self.use_mgd = use_mgd
        self.mgd_alpha = mgd_alpha
        self.mgd_lambda = mgd_lambda
        embed_dim = self.pts_bbox_head.embed_dims

        if use_aux_seg:
            # Lightweight head: 3x3 Conv-GN-GELU + 1x1 Conv → (num_cls+1) channels.
            self.aux_seg_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, aux_seg_num_classes + 1, kernel_size=1, bias=True),
            )

        if use_proj:
            self.bev_distill_proj = BEVDistillProjector(embed_dim, embed_dim, embed_dim)

        if use_mgd:
            # Lightweight align: 1x1 conv for channel remapping (student -> teacher
            # self.mgd_align = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
            self.mgd_align = None
            self.mgd_generation = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                # nn.GroupNorm(32, embed_dim),
                # nn.GELU(),
                # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True),
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
            
        
            segmaps = torch.stack(kwargs['seg_maps'], dim=0)

            depth_maps = None
            if 'depth_maps' in kwargs.keys():
                depth_maps = torch.stack(kwargs['depth_maps'], dim=0)

            noisy_bev = bev_target.detach()
            noisy_bev = noisy_bev.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
            teacher_bev = bev_diffuser(noisy_bev, img_metas, segmaps, depth_maps, grad_fn=get_classifier_gradient)
            # seg_bev_prob = seg_bev_prob.detach().float()  # (B, num_cls+1, H, W)

            B = bev.shape[0]
            C = bev.shape[-1]
            H, W = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
            # teacher_bev: (B, C, H, W) — keep 2D for MGD; flatten inline when needed
            student_bev = bev.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # (B, C, H, W)

            if self.use_mgd:
                if self.mgd_align is not None:
                    preds_S = self.mgd_align(student_bev) 
                else:
                    preds_S = student_bev
                mat = torch.rand((B, 1, H, W), device=preds_S.device)
                mat = torch.where(mat > 1 - self.mgd_lambda, torch.zeros_like(mat), torch.ones_like(mat))
                new_bev = self.mgd_generation(torch.mul(preds_S, mat))
                loss_bev = F.mse_loss(new_bev, teacher_bev.detach().float(), reduction='mean')
                losses['loss_bev'] = loss_bev * self.mgd_alpha

            elif self.use_proj:
                bev_s_2d = self.bev_distill_proj(student_bev.float())
                bev_s_flat = bev_s_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)
                teacher_bev_flat = teacher_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)
                loss_bev = F.mse_loss(bev_s_flat, teacher_bev_flat.detach().float(), reduction="mean")
                losses['loss_bev'] = loss_bev * 100
                
            else:
                teacher_bev = teacher_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)
                loss_bev = F.mse_loss(bev.float(), teacher_bev.detach().float(), reduction="mean")
                losses['loss_bev'] = loss_bev * 100

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