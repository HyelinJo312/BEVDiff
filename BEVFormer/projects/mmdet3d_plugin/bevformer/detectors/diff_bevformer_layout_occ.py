"""Direct-MSE Stage-2 distillation for GT-layout BEVDiffuser Occ3D."""

import copy
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

from .bevformer_occ import BEVFormerOcc
from projects.bevdiffuser.bev_frame_alignment import TeacherBEVAlignment, masked_bev_mse


@DETECTORS.register_module()
class DiffBEVFormerLayoutOcc(BEVFormerOcc):
    """Native BEVFormerOcc with a frozen layout-diffusion teacher."""

    def __init__(self, *args, distill_loss_weight=100.0,
                 teacher_bev_alignment=None, **kwargs):
        super().__init__(*args, **kwargs)
        if distill_loss_weight < 0:
            raise ValueError('distill_loss_weight must be non-negative')
        self.distill_loss_weight = float(distill_loss_weight)
        self.teacher_bev_alignment = (
            TeacherBEVAlignment(**teacher_bev_alignment)
            if teacher_bev_alignment is not None else None)
        self.loss_mix_weight = 0.5
        self._gradient_monitor_context = None
        self._gradient_monitor_pending = None

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        was_training = self.training
        bev = super().obtain_history_bev(imgs_queue, img_metas_list)
        self.train(was_training)
        return bev

    def forward(self, return_loss=True, only_bev=False, **kwargs):
        if only_bev:
            if return_loss:
                raise ValueError('only_bev is an inference-only interface')
            img = kwargs.get('img')
            if not isinstance(img, torch.Tensor):
                raise TypeError(
                    'Stage-2 clean-target extraction expects a temporal image tensor')
            return self.obtain_history_bev(
                img, copy.deepcopy(kwargs['img_metas']))
        return super().forward(return_loss=return_loss, **kwargs)

    def train_step(self, data, optimizer, model_target=None,
                   bev_diffuser=None, progress=None):
        try:
            losses = self(
                **data, model_target=model_target, bev_diffuser=bev_diffuser)
            loss, log_vars = self._parse_losses_mix(
                losses, weight=self.loss_mix_weight)
            return dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(data['img_metas']))
        finally:
            self._gradient_monitor_context = None

    @staticmethod
    def _stack_condition(value, device):
        if isinstance(value, (list, tuple)):
            value = torch.stack(value, dim=0)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f'Diffusion condition must be a tensor, got {type(value)}')
        return value.to(device=device, non_blocking=True)

    def _bev_to_map(self, bev, size=None):
        height, width = size or (self.pts_bbox_head.bev_h,
                                 self.pts_bbox_head.bev_w)
        channels = self.pts_bbox_head.embed_dims
        if bev.ndim == 4 and tuple(bev.shape[1:]) == (
                channels, height, width):
            return bev.contiguous()
        if bev.ndim == 3 and tuple(bev.shape[1:]) == (
                height * width, channels):
            return bev.transpose(1, 2).reshape(
                -1, channels, height, width).contiguous()
        if (bev.ndim == 3 and bev.shape[0] == height * width
                and bev.shape[2] == channels):
            return bev.permute(1, 2, 0).reshape(
                -1, channels, height, width).contiguous()
        raise ValueError(
            f'Unexpected occupancy BEV shape: {tuple(bev.shape)}')

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                      gt_labels_3d=None, voxel_semantics=None,
                      mask_lidar=None, mask_camera=None, gt_labels=None,
                      gt_bboxes=None, img=None, proposals=None,
                      gt_bboxes_ignore=None, img_depth=None, img_mask=None,
                      given_bev=None, model_target=None, bev_diffuser=None,
                      **kwargs):
        if voxel_semantics is None:
            raise ValueError('DiffBEVFormerLayoutOcc requires voxel_semantics')

        bev_target = None
        if bev_diffuser is not None:
            if model_target is None:
                raise ValueError(
                    'A frozen detection model_target is required')
            model_target.eval()
            try:
                with torch.no_grad():
                    bev_target = model_target(
                        return_loss=False,
                        only_bev=True,
                        img=img,
                        img_metas=copy.deepcopy(img_metas)).detach()
            finally:
                model_target.eval()

        queue_length = img.size(1)
        previous_images = img[:, :-1, ...]
        current_image = img[:, -1, ...]
        previous_bev = self.obtain_history_bev(
            previous_images, copy.deepcopy(img_metas))

        current_metas = [each[queue_length - 1] for each in img_metas]
        if not current_metas[0]['prev_bev_exists']:
            previous_bev = None
        image_features = self.extract_feat(
            img=current_image, img_metas=current_metas)
        return self.forward_pts_train(
            image_features,
            gt_bboxes_3d,
            gt_labels_3d,
            voxel_semantics,
            mask_camera,
            current_metas,
            gt_bboxes_ignore,
            previous_bev,
            model_target=model_target,
            bev_diffuser=bev_diffuser,
            bev_target=bev_target,
            **kwargs)

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
                          bev_target=None,
                          **kwargs):
        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True)
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, given_bev=bev)
        losses = self.pts_bbox_head.loss(
            voxel_semantics, mask_camera, outs, img_metas=img_metas)

        if bev_diffuser is not None:
            if bev_target is None:
                raise ValueError('Missing clean detection BEV target')

            def get_condition():
                cond = {}
                if 'layout_obj_classes' in kwargs:
                    cond['obj_class'] = self._stack_condition(kwargs['layout_obj_classes'], bev_.device)
                if 'layout_obj_bboxes' in kwargs:
                    cond['obj_bbox'] = self._stack_condition(kwargs['layout_obj_bboxes'], bev_.device)
                if 'layout_obj_is_valid' in kwargs:
                    cond['is_valid_obj'] = self._stack_condition(kwargs['layout_obj_is_valid'], bev_.device)
                return cond

            alignment = getattr(self, 'teacher_bev_alignment', None)
            teacher_size = (alignment.teacher_bev_size if alignment is not None
                            else (self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w))
            bev_ = self._bev_to_map(bev_target.detach(), teacher_size)
            if alignment is not None:
                teacher_bev = bev_diffuser(bev_, get_condition(), grad_fn=None)
                teacher_support = teacher_bev.new_ones((teacher_bev.shape[0], 1, *teacher_size), dtype=torch.bool)
                teacher_bev, valid_mask = alignment.to_student(
                    teacher_bev, teacher_support, img_metas,
                    (self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w))
            else:
                teacher_bev = bev_diffuser(bev_, get_condition(), grad_fn=None)

            student_bev = self._bev_to_map(bev)
            if alignment is not None and alignment.mask_invalid:
                loss_bev = masked_bev_mse(student_bev, teacher_bev, valid_mask)
            else:
                loss_bev = F.mse_loss(student_bev.float(), teacher_bev.detach().float(), reduction='mean')
            losses['loss_bev'] = loss_bev * self.distill_loss_weight

            if self._gradient_monitor_context is not None:
                from projects.mmdet3d_plugin.models.hooks.occ_gradient_monitor import gradient_sufficient_stats
                task_terms = []
                for name, value in losses.items():
                    if 'loss' in name and 'bev' not in name:
                        task_terms.append(sum(item.mean() for item in value)
                                          if isinstance(value, list) else value.mean())
                task_gradient = torch.autograd.grad(
                    (1.0 - self.loss_mix_weight) * sum(task_terms), bev,
                    retain_graph=True, create_graph=False, allow_unused=True)[0]
                kd_gradient = torch.autograd.grad(
                    self.loss_mix_weight * losses['loss_bev'].mean(), bev,
                    retain_graph=True, create_graph=False, allow_unused=True)[0]
                self._gradient_monitor_pending = gradient_sufficient_stats(
                    self._bev_to_map(task_gradient) if task_gradient is not None else None,
                    self._bev_to_map(kd_gradient) if kd_gradient is not None else None,
                    self._bev_to_map(bev))
                del task_gradient, kd_gradient
        return losses

    def _parse_losses_mix(self, losses, weight=0.5):
        log_vars = OrderedDict()
        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                log_vars[name] = value.mean()
            elif isinstance(value, list):
                log_vars[name] = sum(item.mean() for item in value)
            else:
                raise TypeError(
                    f'{name} is not a tensor or list of tensors')

        total_loss = sum(
            value for name, value in log_vars.items() if 'loss' in name)
        bev_loss = sum(
            value for name, value in log_vars.items()
            if 'loss' in name and 'bev' in name)
        task_loss = total_loss - bev_loss
        loss = (1.0 - weight) * task_loss + weight * bev_loss
        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss

        for name, value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
            log_vars[name] = value.item()
        return loss, log_vars
