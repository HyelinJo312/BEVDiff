# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
from .bevformer import BEVFormer
from projects.bevdiffuser.model_utils import build_unet_v2
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler


class OneStepDiffusionStudent(nn.Module):
    """Trainable Stage-2 diffusion branch initialized from Stage-1 weights."""

    def __init__(self,
                 unet_cfg,
                 unet_checkpoint_dir=None,
                 pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
                 prediction_type=None):
        super().__init__()
        self.noise_scheduler = DDIMGuidedScheduler.from_pretrained(
                                pretrained_model_name_or_path, subfolder="scheduler")
        if prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=prediction_type)

        self.unet = build_unet_v2(unet_cfg)
        if unet_checkpoint_dir is not None:
            self.unet.from_pretrained(unet_checkpoint_dir, subfolder="unet")

    @staticmethod
    def get_segmaps_uncond(segmaps):
        return torch.zeros_like(segmaps)

    def predict_x0(self, x, timesteps, img_metas, segmaps, depth_maps=None):
        seg_bev_cond = self.unet.encode_seg(segmaps, img_metas, depth_maps=depth_maps)
        timesteps = self._format_timesteps(timesteps, x.shape[0], x.device)
        model_output = self.unet(x, timesteps, img_metas, None, depth_maps=depth_maps, seg_bev_maps=seg_bev_cond)
        return self._model_output_to_x0(model_output, x, timesteps)

    @staticmethod
    def _format_timesteps(timesteps, batch_size, device):
        if torch.is_tensor(timesteps):
            timesteps = timesteps.to(device=device, dtype=torch.long)
            if timesteps.ndim == 0:
                timesteps = timesteps.expand(batch_size)
            return timesteps
        return torch.full((batch_size,), int(timesteps), dtype=torch.long, device=device)

    def _model_output_to_x0(self, model_output, sample, timesteps):
        timesteps = self._format_timesteps(timesteps, sample.shape[0], sample.device)
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = self.noise_scheduler.config.prediction_type
        
        if prediction_type == "sample":
            return model_output
        if prediction_type == "epsilon":
            return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if prediction_type == "v_prediction":
            return alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        raise ValueError(
            f"Unsupported prediction_type={prediction_type}. Expected one of "
            "sample, epsilon, or v_prediction.")


@DETECTORS.register_module()
class OneStepDiffBEVFormerSeg(BEVFormer):
    """Stage-2 BEVFormer with a trainable one-step diffusion refinement block."""

    def __init__(self, *args,
                 student_diffuser_cfg=None,
                 one_step_timestep=100,
                 teacher_num_inference_steps=5,
                 distill_loss_weight=1.0,
                 distill_warmup_epochs=0,
                 total_epochs=24,
                 eval_noise_mode='random',
                 train_timestep_mode='fixed',
                 train_timestep_min=1,
                 train_timestep_max=200,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert student_diffuser_cfg is not None, (
            "OneStepDiffBEVFormerSeg requires student_diffuser_cfg.")

        self.student_diffuser = OneStepDiffusionStudent(**student_diffuser_cfg)
        self.one_step_timestep = int(one_step_timestep)
        assert self.one_step_timestep > 0
        self.teacher_num_inference_steps = int(teacher_num_inference_steps)
        self.distill_loss_weight = distill_loss_weight
        self.distill_warmup_epochs = distill_warmup_epochs
        self.total_epochs = total_epochs
        self.eval_noise_mode = eval_noise_mode
        self.train_timestep_mode = train_timestep_mode
        self.train_timestep_min = int(train_timestep_min)
        self.train_timestep_max = int(train_timestep_max)
        assert self.train_timestep_mode in ['fixed', 'random']
        assert self.train_timestep_min > 0
        assert self.train_timestep_min <= self.train_timestep_max
        self._train_progress = None

    def train_step(self, data, optimizer, model_target=None, bev_diffuser=None,
                   progress=None):
        self._train_progress = progress
        losses = self(**data, model_target=model_target, bev_diffuser=bev_diffuser)
        loss, log_vars = self._parse_losses_onestep(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        self._train_progress = None
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
                      **kwargs):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        return self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas,
                                        gt_bboxes_ignore, prev_bev, given_bev, model_target,
                                        bev_diffuser, **kwargs)

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
                          **kwargs):
        # target BEV feature
        bev = self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=True)
        
        # diffusion conditions
        segmaps = self._stack_condition(kwargs, 'seg_maps')
        depth_maps = self._stack_condition(kwargs, 'depth_maps', required=False)

        # one-step diffusion
        student_bev_2d, teacher_bev_2d = self._run_onestep_diffusion(
            bev, img_metas, segmaps, depth_maps, bev_diffuser, training=True)
        student_bev = self._bev_2d_to_flat(student_bev_2d)

        # task head
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, given_bev=student_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        loss_distill = F.mse_loss(student_bev_2d.float(), teacher_bev_2d.detach().float(), reduction='mean')
        losses['loss_diff'] = loss_distill * self._distill_weight()
        return losses

    def simple_test(self, img_metas, img=None, prev_bev=None, given_bev=None,
                    rescale=False, **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]

        bev = self.pts_bbox_head(
            img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        segmaps = self._stack_condition(kwargs, 'seg_maps', required=False)
        student_bev_2d, _ = self._run_onestep_diffusion(
            bev, img_metas, segmaps, depth_maps=None, bev_diffuser=None,
            training=False)
        student_bev = self._bev_2d_to_flat(student_bev_2d)

        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, img_metas, prev_bev, student_bev, rescale=rescale)
        
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

    def _run_onestep_diffusion(self, bev, img_metas, segmaps, depth_maps, bev_diffuser=None, training=True):
        bev_2d = self._bev_flat_to_2d(bev)
        noise = self._make_noise(bev_2d, training=training)
        t_batch = self._make_timestep_batch(bev_2d.shape[0], bev_2d.device, training=training)
        noisy_bev = self.student_diffuser.noise_scheduler.add_noise(bev_2d, noise, t_batch)

        student_segmaps = self._make_student_null_segmaps(segmaps, img_metas, bev_2d.device)
        student_bev = self.student_diffuser.predict_x0(noisy_bev, t_batch, img_metas,
                                                       student_segmaps, depth_maps=None)
        if not training:
            return student_bev, None
        assert bev_diffuser is not None, ("One-step training requires a frozen multi-step bev_diffuser " "teacher.")
        
        with torch.no_grad():
            teacher_bev = self._teacher_multistep_denoise(bev_diffuser, noisy_bev.detach(), img_metas, segmaps, depth_maps, timesteps=t_batch)
            
        return student_bev, teacher_bev.detach()

    def _teacher_multistep_denoise(self, bev_diffuser, noisy_bev, img_metas,
                                   segmaps, depth_maps=None, grad_fn=None,
                                   timesteps=None):
        """Run the frozen teacher from the shared noisy BEV without re-noising."""
        assert bev_diffuser.noise_timesteps == 0, (
            "One-step teacher must not add noise internally")
        if timesteps is None:
            denoise_timestep = self.one_step_timestep
        else:
            assert torch.all(timesteps == timesteps[0]), (
                "One-step teacher expects one shared timestep per batch.")
            denoise_timestep = int(timesteps[0].item())

        num_steps = min(self.teacher_num_inference_steps, denoise_timestep)
        if num_steps <= 0:
            return noisy_bev

        seg_uncond = bev_diffuser.get_segmaps_uncond(segmaps)
        seg_bev_cond = bev_diffuser.unet.encode_seg(
            segmaps, img_metas, depth_maps=depth_maps)
        seg_bev_uncond = bev_diffuser.unet.encode_seg(
            seg_uncond, img_metas, depth_maps=depth_maps)

        timestep_pairs = self._make_teacher_timestep_pairs(denoise_timestep, num_steps, noisy_bev.device)
        x = noisy_bev
        for timestep, prev_timestep in timestep_pairs:
            t_batch = torch.full((x.shape[0],), int(timestep.item()), dtype=torch.long, device=x.device)
            noise_pred_uncond = bev_diffuser.unet(x, t_batch, img_metas, None, depth_maps=depth_maps, seg_bev_maps=seg_bev_uncond)
            noise_pred_cond = bev_diffuser.unet(x, t_batch, img_metas, None, depth_maps=depth_maps, seg_bev_maps=seg_bev_cond)
            noise_pred = noise_pred_uncond + 2 * (noise_pred_cond - noise_pred_uncond)
            classifier_gradient = (grad_fn(x) if bev_diffuser.use_classifier_guidence and grad_fn else None)
            x = self._ddim_step_to_prev(bev_diffuser.noise_scheduler, noise_pred,
                                        int(timestep.item()), int(prev_timestep.item()), x,
                                        classifier_gradient=classifier_gradient)
        return x

    @staticmethod
    def _make_teacher_timestep_pairs(denoise_timestep, num_steps, device):
        timesteps = torch.linspace(denoise_timestep, 0, num_steps + 1, device=device)
        timesteps = timesteps.round().long()
        timesteps = torch.unique_consecutive(timesteps)
        if timesteps[-1].item() != 0:
            timesteps = torch.cat([
                timesteps, torch.zeros(1, dtype=torch.long, device=device)])
        return list(zip(timesteps[:-1], timesteps[1:]))

    @staticmethod
    def _ddim_step_to_prev(scheduler, model_output, timestep, prev_timestep,
                           sample, classifier_gradient=None):
        alpha_prod_t = scheduler.alphas_cumprod[timestep].to(
            device=sample.device, dtype=sample.dtype)
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep].to(
            device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = scheduler.config.prediction_type

        if prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
            pred_epsilon = model_output
        elif prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t.sqrt() * pred_original_sample
            ) / beta_prod_t.sqrt()
        elif prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t.sqrt() * sample
                - beta_prod_t.sqrt() * model_output)
            pred_epsilon = (
                alpha_prod_t.sqrt() * model_output
                + beta_prod_t.sqrt() * sample)
        else:
            raise ValueError(
                f"Unsupported prediction_type={prediction_type}. Expected "
                "one of sample, epsilon, or v_prediction.")

        if classifier_gradient is not None:
            pred_epsilon = pred_epsilon - beta_prod_t.sqrt() * classifier_gradient
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * pred_epsilon
            ) / alpha_prod_t.sqrt()

        if scheduler.config.thresholding:
            pred_original_sample = scheduler._threshold_sample(
                pred_original_sample)
        elif scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -scheduler.config.clip_sample_range,
                scheduler.config.clip_sample_range)

        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * pred_epsilon
        prev_sample = (
            alpha_prod_t_prev.sqrt() * pred_original_sample
            + pred_sample_direction)
        return prev_sample

    def _distill_weight(self):
        if (self.distill_warmup_epochs <= 0 or self._train_progress is None or self.total_epochs <= 0):
            return self.distill_loss_weight
        cur_epoch = float(self._train_progress) * float(self.total_epochs)
        warmup = min(1.0, cur_epoch / float(self.distill_warmup_epochs))
        return self.distill_loss_weight * warmup

    def _parse_losses_onestep(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        distill_loss = sum(_value for _key, _value in log_vars.items()
                           if 'loss' in _key and 'diff' in _key)
        task_loss = sum(_value for _key, _value in log_vars.items()
                        if 'loss' in _key and 'diff' not in _key)
        loss = task_loss + distill_loss

        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss
        log_vars['distill_loss'] = distill_loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def _bev_flat_to_2d(self, bev):
        b, hw, c = bev.shape
        h, w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        assert hw == h * w
        return bev.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

    @staticmethod
    def _bev_2d_to_flat(bev):
        b, c, h, w = bev.shape
        return bev.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()

    def _make_timestep_batch(self, batch_size, device, training=True):
        if training and self.train_timestep_mode == 'random':
            timestep = torch.randint(
                self.train_timestep_min,
                self.train_timestep_max + 1,
                (1,), dtype=torch.long, device=device)
            return timestep.repeat(batch_size)
        return torch.full((batch_size,), self.one_step_timestep, dtype=torch.long, device=device)

    def _make_student_null_segmaps(self, segmaps, img_metas, device):
        if segmaps is not None:
            return torch.zeros_like(segmaps, device=device)

        first_meta = img_metas[0]
        if 'pad_shape' in first_meta:
            frame_shapes = first_meta['pad_shape']
        elif 'img_shape' in first_meta:
            frame_shapes = first_meta['img_shape']
        else:
            raise KeyError(
                'img_metas must include pad_shape or img_shape to build '
                'condition-free student null segmaps.')

        num_views = len(frame_shapes)
        frame_h, frame_w = frame_shapes[0][:2]
        return torch.zeros(
            (len(img_metas), num_views, frame_h, frame_w),
            dtype=torch.long,
            device=device)

    def _make_noise(self, x, training=True):
        if training or self.eval_noise_mode == 'random':
            return torch.randn_like(x)
        if self.eval_noise_mode == 'zero':
            return torch.zeros_like(x)
        if self.eval_noise_mode == 'fixed':
            generator = torch.Generator(device=x.device)
            generator.manual_seed(0)
            return torch.randn(
                x.shape, generator=generator, device=x.device, dtype=x.dtype)
        raise ValueError(f'Unknown eval_noise_mode={self.eval_noise_mode}')

    @staticmethod
    def _stack_condition(kwargs, key, required=True):
        value = kwargs.get(key, None)
        if value is None:
            if required:
                raise KeyError(f'{key} is required for one-step diffusion.')
            return None
        if isinstance(value, torch.Tensor):
            return value
        return torch.stack(value, dim=0)
