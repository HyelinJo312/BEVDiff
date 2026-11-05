"""Stage-1 GT-layout BEV diffusion training for the native Occ3D baseline."""

import argparse
import logging
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import accelerate
import datasets
import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info, load_checkpoint, save_checkpoint
from mmdet.apis import set_random_seed
from mmdet.utils import setup_multi_processes
from mmdet3d.datasets import build_dataset
from packaging import version
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

torch.backends.cudnn.enabled = True
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

from projects.bevdiffuser.model_utils import build_unet
from projects.mmdet3d_plugin.bevformer.detectors.bevformer_occ import BEVFormerOcc
from projects.bevdiffuser.occ_diffusion_utils import (
    get_occ_model,
    load_diffusion_weights,
    occ_task_parameters,
    set_occ_task_train,
)
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

logger = get_logger(__name__, log_level='INFO')
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _container_data(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError('Only single-scale, non-flipped batches are supported')
        value = value[0]
    if isinstance(value, DataContainer):
        value = value.data[0]
    return value


def _batch_tensor(batch, key, device):
    value = _container_data(batch[key])
    if isinstance(value, (list, tuple)):
        value = torch.stack(value, dim=0)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'batch[{key!r}] must contain tensors, got {type(value)}')
    return value.to(device=device, non_blocking=True)


def _current_img_metas(batch):
    metas = _container_data(batch['img_metas'])
    current = []
    for sample_metas in metas:
        if (isinstance(sample_metas, dict) and sample_metas
                and all(isinstance(key, int) for key in sample_metas)):
            sample_metas = sample_metas[max(sample_metas)]
        current.append(sample_metas)
    return current


def _bev_tokens_to_map(bev, cfg):
    channels, height, width = cfg._dim_, cfg.bev_h_, cfg.bev_w_
    if bev.ndim == 4 and tuple(bev.shape[1:]) == (channels, height, width):
        return bev.contiguous()
    if bev.ndim == 3 and tuple(bev.shape[1:]) == (height * width, channels):
        return bev.reshape(-1, height, width, channels).permute(0, 3, 1, 2).contiguous()
    if bev.ndim == 3 and bev.shape[0] == height * width and bev.shape[2] == channels:
        return bev.permute(1, 2, 0).reshape(-1, channels, height, width).contiguous()
    raise ValueError(f'Unexpected occupancy BEV shape {tuple(bev.shape)}')


def _bev_map_to_tokens(bev):
    return bev.permute(0, 2, 3, 1).reshape(bev.shape[0], -1, bev.shape[1]).contiguous()


def _layout_condition(batch, device, unet, uncond_probability):
    condition = {
        'obj_class': _batch_tensor(batch, 'layout_obj_classes', device),
        'obj_bbox': _batch_tensor(batch, 'layout_obj_bboxes', device),
        'is_valid_obj': _batch_tensor(batch, 'layout_obj_is_valid', device),
    }
    if uncond_probability <= 0 or torch.rand((), device=device) >= uncond_probability:
        return condition

    encoder = _unwrap(unet).layout_encoder
    condition['obj_class'].fill_(encoder.num_classes_for_layout_object - 1)
    condition['obj_class'][:, 0] = encoder.num_classes_for_layout_object - 2
    condition['obj_bbox'].zero_()
    condition['obj_bbox'][:, 0] = condition['obj_bbox'].new_tensor(
        [0, 0, 0, 1, 1, 1, 0, 0, 0])
    condition['is_valid_obj'].zero_()
    condition['is_valid_obj'][:, 0] = 1.0
    return condition


def _resolve_checkpoint(args, cfg):
    value = args.bev_checkpoint or cfg.get('stage1_bev_checkpoint')
    if value is None:
        raise ValueError('Set --bev_checkpoint or stage1_bev_checkpoint')
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        REPOSITORY_ROOT / path,
    ]
    resolved = next((candidate.resolve() for candidate in candidates
                     if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f'BEVFormerOcc checkpoint not found: {value}')
    args.bev_checkpoint = str(resolved)


def train():
    args = parse_args()
    cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _resolve_checkpoint(args, cfg)
    setup_multi_processes(cfg)

    if args.resume_from_checkpoint:
        args.output_dir = os.path.dirname(
            os.path.normpath(args.resume_from_checkpoint))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None,
        log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, args.logging_dir)),
    )
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='scheduler')
    scheduler.register_to_config(prediction_type=args.prediction_type)

    bev_model = get_occ_model(args, cfg)
    bev_model.requires_grad_(False)
    occ_model = _unwrap(bev_model)
    if not isinstance(occ_model, BEVFormerOcc):
        raise TypeError('layout_base_occ.py must build BEVFormerOcc')
    if args.task_loss_scale:
        for parameter in occ_task_parameters(occ_model):
            parameter.requires_grad_(True)

    unet = build_unet(cfg.unet)
    if args.pretrained_unet_checkpoint:
        load_diffusion_weights(unet, args.pretrained_unet_checkpoint)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if version.parse(accelerate.__version__) < version.parse('0.16.0'):
        raise RuntimeError('accelerate>=0.16.0 is required')

    trained_parameters = [parameter for parameter in unet.parameters()
                          if parameter.requires_grad]
    if args.task_loss_scale:
        trained_parameters += [
            parameter for parameter in occ_task_parameters(occ_model)
            if parameter.requires_grad
        ]
    optimizer = torch.optim.AdamW(
        trained_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)

    dataset_defaults = dict(
        pc_range=cfg.point_cloud_range,
        use_3d_bbox=cfg.use_3d_bbox,
        num_classes=cfg.num_classes,
        num_bboxes=cfg.num_bboxes)
    with accelerator.main_process_first():
        train_dataset = build_dataset(
            cfg.data.train, default_args=dataset_defaults)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=args.train_batch_size,
        workers_per_gpu=args.dataloader_num_workers,
        num_gpus=get_dist_info()[1],
        dist=(args.launcher != 'none'),
        seed=args.seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    def save_model_hook(models, weights, output_dir):
        while models:
            model = accelerator.unwrap_model(models.pop())
            if accelerator.is_main_process:
                model.save_pretrained(os.path.join(output_dir, 'unet'))
            weights.pop()

    def load_model_hook(models, input_dir):
        while models:
            model = accelerator.unwrap_model(models.pop())
            load_diffusion_weights(model, input_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler)
    bev_model.to(accelerator.device, dtype=torch.float32)

    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / updates_per_epoch)
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        occ_path = os.path.join(args.resume_from_checkpoint, 'occ_model.pth')
        if args.task_loss_scale and os.path.isfile(occ_path):
            load_checkpoint(bev_model, occ_path, map_location='cpu', strict=True)
        global_step = int(
            os.path.basename(os.path.normpath(args.resume_from_checkpoint)).split('-')[-1])
        first_epoch = global_step // updates_per_epoch
        resume_step = ((global_step * args.gradient_accumulation_steps)
                       % (updates_per_epoch * args.gradient_accumulation_steps))

    writer = None
    if accelerator.is_main_process and args.report_to == 'tensorboard':
        writer = SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))
    accelerator.init_trackers(
        project_name=args.tracker_project_name,
        config=dict(vars(args)),
        init_kwargs={'wandb': {'name': args.tracker_run_name}}
        if args.report_to == 'wandb' else {})

    logger.info('***** Running GT-layout Occ3D Stage 1 *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    progress = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        ncols=100)
    progress.set_description('Steps')

    for epoch in range(first_epoch, num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        unet.train()
        for step, batch in enumerate(train_loader):
            if (args.resume_from_checkpoint and epoch == first_epoch
                    and step < resume_step):
                continue
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    clean_bev = bev_model(
                        return_loss=False, only_bev=True, **batch).detach()
                clean_bev = _bev_tokens_to_map(clean_bev, cfg)
                set_occ_task_train(occ_model, bool(args.task_loss_scale))

                noise = torch.randn_like(clean_bev)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (clean_bev.shape[0],), device=clean_bev.device).long()
                noisy_bev = scheduler.add_noise(clean_bev, noise, timesteps)
                if args.prediction_type == 'sample':
                    target = clean_bev
                elif args.prediction_type == 'epsilon':
                    target = noise
                else:
                    target = scheduler.get_velocity(clean_bev, noise, timesteps)

                condition = _layout_condition(
                    batch, clean_bev.device, unet, args.uncond_prob)
                prediction = unet(noisy_bev, timesteps, **condition)[0]
                denoise_loss = F.mse_loss(
                    prediction.float(), target.float(), reduction='mean')

                if args.task_loss_scale and args.prediction_type == 'sample':
                    task_losses = bev_model(
                        return_loss=True,
                        given_bev=_bev_map_to_tokens(prediction),
                        voxel_semantics=_batch_tensor(
                            batch, 'voxel_semantics', prediction.device),
                        mask_camera=_batch_tensor(
                            batch, 'mask_camera', prediction.device),
                        mask_lidar=_batch_tensor(
                            batch, 'mask_lidar', prediction.device),
                        img_metas=_current_img_metas(batch))
                    task_loss, _ = occ_model._parse_losses(task_losses)
                else:
                    task_loss = denoise_loss.new_zeros(())
                loss = denoise_loss + args.task_loss_scale * task_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trained_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                metrics = {
                    'train/denoise_loss': denoise_loss.detach().item(),
                    'train/occ_task_loss': task_loss.detach().item(),
                    'train/total_loss': loss.detach().item(),
                    'lr/learning_rate': lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(metrics, step=global_step)
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, global_step)
                progress.set_postfix(
                    loss=loss.detach().item(),
                    lr=lr_scheduler.get_last_lr()[0],
                    epoch=epoch)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        checkpoints = [
                            name for name in os.listdir(args.output_dir)
                            if name.startswith('checkpoint-')
                        ]
                        checkpoints.sort(
                            key=lambda name: int(name.split('-')[-1]))
                        if args.checkpoints_total_limit is not None:
                            excess = (len(checkpoints)
                                      - args.checkpoints_total_limit + 1)
                            for name in checkpoints[:max(0, excess)]:
                                shutil.rmtree(
                                    os.path.join(args.output_dir, name))
                    accelerator.wait_for_everyone()
                    checkpoint_dir = os.path.join(
                        args.output_dir, f'checkpoint-{global_step}')
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process and args.task_loss_scale:
                        save_checkpoint(
                            bev_model,
                            os.path.join(checkpoint_dir, 'occ_model.pth'))
                    accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    if writer is not None:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GT-layout BEVDiffuser for Occ3D')
    parser.add_argument('--bev_config', required=True)
    parser.add_argument('--bev_checkpoint', default=None)
    parser.add_argument('--pretrained_unet_checkpoint', default=None)
    parser.add_argument('--pretrained_model_name_or_path',
                        default='stabilityai/stable-diffusion-2-1')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch')
    parser.add_argument('--local_rank', '--local-rank',
                        dest='local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--output_dir', default='results/layout_occ')
    parser.add_argument('--logging_dir', default='logs')
    parser.add_argument('--report_to',
                        choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument('--tracker_project_name',
                        default='BEVDiffuser-Occ3D')
    parser.add_argument('--tracker_run_name',
                        default='BEVDiffuser_layout_base_occ')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--max_train_steps', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', default='constant')
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--uncond_prob', type=float, default=0.2)
    parser.add_argument('--prediction_type',
                        choices=['sample', 'epsilon', 'v_prediction'],
                        default='sample')
    parser.add_argument('--task_loss_scale', type=float, default=0.1)
    parser.add_argument('--checkpointing_steps', type=int, default=10000)
    parser.add_argument('--checkpoints_total_limit', type=int, default=3)
    parser.add_argument('--resume_from_checkpoint', default=None)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot both be specified')
    if args.options:
        warnings.warn('--options is deprecated; use --cfg-options')
        args.cfg_options = args.options
    if not 0 <= args.uncond_prob <= 1:
        raise ValueError('--uncond_prob must be in [0, 1]')
    if args.task_loss_scale < 0:
        raise ValueError('--task_loss_scale must be non-negative')
    return args


if __name__ == '__main__':
    train()
