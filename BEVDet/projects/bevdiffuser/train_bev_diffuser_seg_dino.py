# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Stage 1 semantic diffusion over frozen DINOv3 BEV targets.

Based on train_bev_diffuser_only_seg.py. MMCV owns distributed data sampling;
Accelerate owns UNet/optimizer synchronization. No detection model is built.
"""

import argparse
import importlib
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch
import transformers
import wandb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.enabled = False
# cuDNN 및 convolution 알고리즘 자동 선택
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# FP32 convolution과 행렬 연산에 TF32 사용 허용
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.utils import import_modules_from_strings
from mmdet3d.datasets import build_dataset
from safetensors.torch import load_file

# Support both direct script execution and the existing project import paths.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
sys.path.insert(0, str(SCRIPT_DIR))

# Register the plugin before data_utils, whose datasets depend on its registry.
import projects.mmdet3d_plugin  # noqa: F401, E402
import projects.bevdiffuser.data_utils  # noqa: F401, E402
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from dinov3_bev_extractor import DINOv3BEVExtractor
from layout_diffusion.temporal_dino_condition import TemporalBEVWarper, build_temporal_dino_condition
from model_utils import build_unet_v2

logger = get_logger(__name__, log_level="INFO")


def unpack_batch(batch, device):
    """Unwrap one MMCV device group and prepare current or temporal DINO inputs."""

    def payload(key):
        value = batch[key]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError('Only one test augmentation is supported')
            value = value[0]
        if not isinstance(value, DataContainer) or len(value.data) != 1:
            raise ValueError(f'{key} must contain one collated MMCV device group')
        return value.data[0]

    def tensor(key):
        value = payload(key)
        if isinstance(value, (list, tuple)):
            value = torch.stack(value)
        return value.to(device, non_blocking=True)

    images = tensor('img')
    metas = payload('img_metas')
    temporal_valid = None
    if 'temporal_depth_maps' in batch:
        depth = tensor('temporal_depth_maps')
        temporal_valid = tensor('temporal_valid_mask')
    else:
        if images.ndim == 6:
            images = images[:, -1]
            metas = [meta if 'filename' in meta else meta[max(meta)] for meta in metas]
        depth = tensor('depth_maps')

    if 'semantic_bev_probabilities' in batch:
        condition = dict(
            seg_cond=None,
            semantic_prob=tensor('semantic_bev_probabilities'))
    else:
        current_depth = depth[:, -1] if depth.ndim == 5 else depth
        condition = dict(seg_cond=tensor('seg_maps'), depth_maps=current_depth)
    return images, metas, depth, temporal_valid, condition


def attach_cls_condition(condition, dino_result, use_global_cls):
    """Reuse the frozen extractor CLS output for both training and validation."""
    condition = dict(condition)
    if use_global_cls:
        condition['cls_tokens'] = dino_result['cls_tokens']
    return condition


def drop_condition(condition, probability):
    """Drop semantic, CLS and temporal conditions with one sample mask."""
    condition = dict(condition)
    key = 'semantic_prob' if 'semantic_prob' in condition else 'seg_cond'
    value = condition[key]
    keep = torch.rand(value.shape[0], device=value.device) >= probability
    condition[key] = value * keep.reshape(
        -1, *([1] * (value.ndim - 1))).to(value.dtype)
    if 'cls_tokens' in condition:
        condition['cls_keep_mask'] = keep
    if 'temporal_bevs' in condition:
        condition['temporal_keep_mask'] = keep
    return condition


def diffusion_target(scheduler, latents, noise, timesteps):
    """Construct the target for the scheduler's configured prediction type."""
    prediction_type = scheduler.config.prediction_type
    if prediction_type == 'sample':
        return latents
    if prediction_type == 'epsilon':
        return noise
    if prediction_type == 'v_prediction':
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f'Unsupported prediction type: {prediction_type}')


def predict_clean(scheduler, prediction, noisy, timesteps):
    """Convert a model prediction to an unclipped clean BEV estimate."""
    kind = scheduler.config.prediction_type
    if kind == 'sample':
        return prediction
    alpha = scheduler.alphas_cumprod.to(noisy.device)[timesteps].reshape(-1, 1, 1, 1)
    if kind == 'epsilon':
        return (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
    if kind == 'v_prediction':
        return alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction
    raise ValueError(kind)


def save_rng_state(path, device, loader):
    """Use only tensors/builtins so PyTorch weights_only loading also works.

    Accelerate 0.20 silently skips its NumPy RNG pickle on PyTorch >= 2.6.
    Keep an explicit per-rank RNG checkpoint and fail if it cannot be restored.
    """
    numpy_state = np.random.get_state()
    state = dict(
        python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        torch=torch.get_rng_state(),
        dataloader=loader.generator.get_state(),
    )
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    torch.save(state, path)


def load_rng_state(path, device, loader):
    state = torch.load(path, map_location='cpu', weights_only=True)
    random.setstate(state['python'])
    name, keys, position, has_gauss, cached_gaussian = state['numpy']
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )
    torch.set_rng_state(state['torch'])
    loader.generator.set_state(state['dataloader'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def load_unet_strict(unet, path):
    path = Path(path)
    candidates = (
        [path]
        if path.is_file()
        else [path / SAFETENSORS_WEIGHTS_NAME, path / 'unet' / SAFETENSORS_WEIGHTS_NAME]
    )
    weights = next((p for p in candidates if p.is_file()), None)
    if weights is None:
        raise FileNotFoundError(f'No UNet safetensors checkpoint under {path}')
    # Never partially load an old 256-channel teacher into this 384-channel UNet.
    unet.load_state_dict(load_file(str(weights), device='cpu'), strict=True)


@torch.no_grad()
def validate(unet, extractor, loader, scheduler, device, max_batches, timestep, seed,
             use_global_cls=False, temporal_warper=None):
    """Fixed-noise x0 reconstruction metrics; no detection supervision or head."""
    was_training = unet.training
    unet.eval()
    totals = torch.zeros(8, device=device, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            images, metas, depth, temporal_valid, condition = unpack_batch(
                batch, device)
            if temporal_warper is None:
                dino_result = extractor(images, metas, depth)
            else:
                dino_result, metas, history_bevs, history_masks = (
                    build_temporal_dino_condition(
                        extractor, temporal_warper, images, depth,
                        metas, temporal_valid))
                condition['temporal_bevs'] = history_bevs
                condition['temporal_masks'] = history_masks
                condition['temporal_keep_mask'] = torch.ones(
                    images.shape[0], device=device, dtype=torch.bool)
            condition = attach_cls_condition(condition, dino_result, use_global_cls)
            target, observed = dino_result['bev_features'], dino_result['observed_mask']
            noise = torch.randn(target.shape, generator=generator, device=device)
            times = torch.full((target.shape[0],), timestep, device=device, dtype=torch.long)
            noisy = scheduler.add_noise(target, noise, times)
            pred = unet(noisy, times, metas, **condition).float()
            clean = predict_clean(scheduler, pred, noisy, times)
            error = (clean - target).square()
            cosine = F.cosine_similarity(clean, target, dim=1)
            obs = observed.expand_as(error)
            totals += torch.stack(
                (
                    error.sum(),
                    error.new_tensor(error.numel()),
                    error.masked_select(obs).sum(),
                    obs.sum(),
                    error.masked_select(~obs).sum(),
                    (~obs).sum(),
                    cosine.masked_select(observed[:, 0]).sum(),
                    observed.sum(),
                )
            ).double()
    finally:
        unet.train(was_training)
    total, count, obs_sum, obs_count, empty_sum, empty_count, cos_sum, cells = totals.tolist()
    return dict(
        mse=total / max(count, 1),
        observed_mse=obs_sum / max(obs_count, 1),
        unobserved_mse=empty_sum / max(empty_count, 1),
        observed_cosine=cos_sum / max(cells, 1),
        observed_fraction=obs_count / max(count, 1),
    )


def build_stage1_dataloader(dataset_cfg, cfg, args, accelerator, training):
    """Build an MMCV loader without applying Accelerate sharding twice."""
    options = dict(dataset_cfg)
    options.pop('samples_per_gpu', None)
    if not training:
        options['load_annos'] = False
        if not cfg.unet.parameters.get('use_temporal_condition', False):
            options['test_mode'] = True
    dataset = build_dataset(options, default_args=dict(pc_range=cfg.point_cloud_range))
    return build_dataloader(
        dataset,
        samples_per_gpu=args.train_batch_size if training else 1,
        workers_per_gpu=args.dataloader_num_workers if training else 0,
        num_gpus=1,
        dist=training,
        shuffle=training,
        seed=args.seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index),
    )


def validate_extractor_config(extractor, cfg):
    """Check that the DINO target and semantic condition use matching geometry."""
    channels = extractor.encoder.hidden_size
    if (
        channels != 384
        or cfg.unet.parameters.in_channels != channels
        or cfg.unet.parameters.out_channels != channels
    ):
        raise ValueError('This Stage 1 requires DINOv3 ViT-S and UNet input/output channels = 384')
    if (cfg.unet.parameters.get('use_global_cls', False)
            and cfg.unet.parameters.get('cls_channels', 384) != channels):
        raise ValueError('CLS channels must match the frozen DINO encoder hidden size')
    if (extractor.projector.bev_h, extractor.projector.bev_w) != (cfg.bev_h_, cfg.bev_w_):
        raise ValueError('DINO and semantic BEV grid sizes must match')
    if tuple(extractor.projector.pc_range) != tuple(cfg.point_cloud_range):
        raise ValueError('DINO and dataset point cloud ranges must match')
    semantic = cfg.unet.parameters.seg_bev_aligner
    if (semantic.bev_h, semantic.bev_w) != (
        extractor.projector.bev_h,
        extractor.projector.bev_w,
    ) or tuple(semantic.pc_range) != tuple(extractor.projector.pc_range):
        raise ValueError('DINO and semantic projection geometry must match')
    if tuple(cfg.data.train.depth_range) != tuple(extractor.projector.depth_range):
        raise ValueError('DINO projector and depth loader ranges must match')
    return channels


def build_teacher_spec(cfg, extractor, noise_scheduler):
    """Return JSON-compatible target settings for checkpoint compatibility checks."""
    spec = dict(
        format_version=1,
        latent_space='frozen_dinov3_metric_depth',
        channels=extractor.encoder.hidden_size,
        extractor=cfg.dino_bev_extractor,
        resolved_revision=extractor.encoder.revision,
        image_mean=extractor.encoder.image_mean.flatten().tolist(),
        image_std=extractor.encoder.image_std.flatten().tolist(),
        image_norm=cfg.img_norm_cfg,
        data_train=cfg.data.train,
        unet=cfg.unet,
        scheduler={k: v for k, v in dict(noise_scheduler.config).items() if not k.startswith('_')},
    )
    return json.loads(json.dumps(spec))


def train():
    """Optimize the semantic UNet with frozen DINO targets and resumable state."""
    args = parse_args()
    resume_output_dir = None
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = str(Path(args.resume_from_checkpoint).resolve())
        args.output_dir = str(Path(args.resume_from_checkpoint).parent)
        resume_output_dir = args.output_dir
    cfg = Config.fromfile(args.bev_config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.gradient_checkpointing:
        cfg.unet.parameters.use_checkpoint = True
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True, broadcast_buffers=False)
        ],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if resume_output_dir is not None:
        logger.info(
            f"resume_from_checkpoint detected, reusing output dir "
            f"{resume_output_dir}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    # Build DINOv3 extractor
    extractor = DINOv3BEVExtractor(**cfg.dino_bev_extractor).to(accelerator.device)
    channels = validate_extractor_config(extractor, cfg)
    use_global_cls = cfg.unet.parameters.get('use_global_cls', False)
    
    use_temporal_condition = cfg.unet.parameters.get('use_temporal_condition', False)
    temporal_warper = None
    if use_temporal_condition:
        temporal_warper = TemporalBEVWarper(cfg.bev_h_, cfg.bev_w_, cfg.point_cloud_range).to(accelerator.device)

    # Build UNet and noise scheduler
    unet = build_unet_v2(cfg.unet)
    if args.pretrained_unet_checkpoint:
        load_unet_strict(unet, args.pretrained_unet_checkpoint)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    noise_scheduler.register_to_config(prediction_type=args.prediction_type, clip_sample=False)

    if not 0 <= args.validation_timestep < noise_scheduler.config.num_train_timesteps:
        raise ValueError('validation_timestep is outside the scheduler training range')

    with accelerator.main_process_first():
        train_dataloader = build_stage1_dataloader(cfg.data.train, cfg, args, accelerator, training=True)

    if len(train_dataloader) == 0:
        raise ValueError('Training dataloader is empty')

    val_loader = (build_stage1_dataloader(cfg.data.val, cfg, args, accelerator, training=False)
        if accelerator.is_main_process and args.validation_batches
        else None)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # MMCV loaders are already sharded; preparing them would shard them twice.
    unet, optimizer = accelerator.prepare(unet, optimizer)
    accelerator.register_for_checkpointing(lr_scheduler)
    set_seed(args.seed + accelerator.process_index)

    spec = build_teacher_spec(cfg, extractor, noise_scheduler)
    topology = dict(
        world_size=accelerator.num_processes,
        batch_size=args.train_batch_size,
        accumulation=args.gradient_accumulation_steps,
        batches_per_epoch=len(train_dataloader),
        seed=args.seed,
    )
    # Match the reference trainer's epoch, tracker and batch-size reporting.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (args.train_batch_size * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    output = Path(args.output_dir)
    tb_writer = None

    # Initialize the same trackers and manual TensorBoard writer as the
    # reference semantic BEV trainer.
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(output / 'config.py'))
        tracker_config = dict(vars(args))
        if (args.report_to == 'tensorboard'
                and isinstance(tracker_config.get('cfg_options'), dict)):
            tracker_config['cfg_options'] = json.dumps(
                tracker_config['cfg_options'], sort_keys=True)

        if args.resume_from_checkpoint:
            resume_ckpt_number = args.resume_from_checkpoint.split('-')[-1]
            args.tracker_run_name = (
                f"{args.tracker_run_name}-resume-{resume_ckpt_number}")

        init_kwargs = {}
        if args.report_to == 'wandb':
            wandb.init(
                project=args.tracker_project_name,
                name=args.tracker_run_name,
                id=args.tracker_run_name,
            )
            init_kwargs = {
                'wandb': {
                    'name': args.tracker_run_name,
                }
            }

        (output / 'run_args.json').write_text(json.dumps(vars(args), indent=2))
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

        if args.report_to == 'tensorboard':
            tb_logdir = os.path.join(args.output_dir, 'tensorboard_logs')
            tb_writer = SummaryWriter(log_dir=tb_logdir)

    def save_hook(models, weights, directory):
        for model in models:
            if accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(str(Path(directory) / 'unet'))
            weights.pop()

    def load_hook(models, directory):
        while models:
            load_unet_strict(accelerator.unwrap_model(models.pop()), Path(directory) / 'unet')

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    global_step, first_epoch, first_batch = 0, 0, 0
    if args.resume_from_checkpoint:
        resume = Path(args.resume_from_checkpoint)
        if json.loads((resume / 'teacher_spec.json').read_text()) != spec:
            raise ValueError(
                'Checkpoint teacher/UNet/scheduler specification differs from this run'
            )
        state = json.loads((resume / 'training_state.json').read_text())
        if state['topology'] != topology:
            raise ValueError(
                'Resume requires identical world size, batch, accumulation, seed and dataloader length'
            )
        accelerator.load_state(str(resume))
        load_rng_state(resume / f'rng_state_{accelerator.process_index}.pt', accelerator.device, train_dataloader)
        global_step, first_epoch, first_batch = (
            state['global_step'],
            state['epoch'],
            state['next_batch'],
        )
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataloader.dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Num update steps per epoch = %d', num_update_steps_per_epoch)
    logger.info('  Instantaneous batch size per device = %d', args.train_batch_size)
    logger.info(
        '  Total train batch size (w. parallel, distributed & accumulation) = %d',
        total_batch_size)
    logger.info('  Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', args.max_train_steps)
    logger.info('  Is SD21: %s', args.pretrained_model_name_or_path == 'stabilityai/stable-diffusion-2-1')
    condition_source = (f'NumPy cache: {cfg.data.train.semantic_bev_cache_root}'
                        if cfg.data.train.get('use_semantic_bev_cache') else 'live SAM3/Metric3D')
    logger.info('  Semantic BEV condition source: %s', condition_source)
    logger.info('  Frozen DINOv3: %s; BEV channels: %d; only UNet optimized',
                extractor.encoder.revision, channels)
    if args.resume_from_checkpoint:
        logger.info('  Resuming from checkpoint: %s', args.resume_from_checkpoint)

    def checkpoint(epoch, next_batch):
        path = output / f'checkpoint-{global_step}'
        accelerator.wait_for_everyone()
        accelerator.save_state(str(path))
        save_rng_state(
            path / f'rng_state_{accelerator.process_index}.pt', accelerator.device, train_dataloader
        )
        if accelerator.is_main_process:
            (path / 'teacher_spec.json').write_text(json.dumps(spec, indent=2))
            (path / 'training_state.json').write_text(
                json.dumps(
                    dict(
                        global_step=global_step,
                        epoch=epoch,
                        next_batch=next_batch,
                        topology=topology,),indent=2,))
            cfg.dump(str(path / 'config.py'))
            noise_scheduler.save_pretrained(str(path / 'scheduler'))
            logger.info(f"Saved state to {path}")
            if val_loader is not None:
                logger.info(
                    f"Evaluating at epoch {epoch} step {global_step}")
                # Validation must not consume the training RNG stream.
                devices = ([accelerator.device.index]
                           if accelerator.device.type == 'cuda' else [])
                with torch.random.fork_rng(devices=devices):
                    metrics = validate(
                        accelerator.unwrap_model(unet),
                        extractor,
                        val_loader,
                        noise_scheduler,
                        accelerator.device,
                        args.validation_batches,
                        args.validation_timestep,
                        args.seed,
                        use_global_cls=use_global_cls,
                        temporal_warper=temporal_warper,
                    )
                (path / 'validation.json').write_text(
                    json.dumps(metrics, indent=2))
                if args.report_to == 'wandb':
                    for name, value in metrics.items():
                        wandb.log(
                            {f'val/{name}': value}, step=step_cnt)
                elif args.report_to == 'tensorboard':
                    for name, value in metrics.items():
                        tb_writer.add_scalar(
                            f'val/{name}', value, global_step=step_cnt)
                logger.info(
                    'Validation at step %d: %s', global_step, metrics)
            checkpoints = sorted(
                (p for p in output.glob('checkpoint-*')
                 if p.is_dir() and p.name[11:].isdigit()),
                key=lambda p: int(p.name[11:]),
            )
            removing_checkpoints = checkpoints[: -args.checkpoints_total_limit]
            if removing_checkpoints:
                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing "
                    f"{len(removing_checkpoints)} checkpoints")
                logger.info(
                    'removing checkpoints: '
                    + ', '.join(path.name for path in removing_checkpoints))
            for old in removing_checkpoints:
                shutil.rmtree(old)
        accelerator.wait_for_everyone()

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description('Steps')
    optimizer.zero_grad()
    epoch = first_epoch
    step_cnt = first_epoch * len(train_dataloader) + first_batch
    train_loss = 0.0

    while global_step < args.max_train_steps:
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        unet.train()
        for batch_index, batch in enumerate(train_dataloader):
            if epoch == first_epoch and batch_index < first_batch:
                continue
            with accelerator.accumulate(unet):
                images, metas, depth, temporal_valid, condition = unpack_batch(
                    batch, accelerator.device)
                if temporal_warper is None:
                    dino_result = extractor(images, metas, depth)
                else:
                    dino_result, metas, history_bevs, history_masks = build_temporal_dino_condition(
                                                                        extractor, temporal_warper, images, depth,
                                                                        metas, temporal_valid)
                    condition['temporal_bevs'] = history_bevs
                    condition['temporal_masks'] = history_masks
                latents = dino_result['bev_features']
                condition = attach_cls_condition(condition, dino_result, use_global_cls)

                if not torch.isfinite(latents).all():
                    raise ValueError('Non-finite DINO BEV target')

                noise = torch.randn_like(latents)
                times = torch.randint(noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy = noise_scheduler.add_noise(latents, noise, times)

                pred = unet(noisy, times, metas, **drop_condition(condition, args.uncond_prob))

                target = diffusion_target(noise_scheduler, latents, noise, times)

                if pred.shape != target.shape:
                    raise ValueError(
                        f'UNet prediction {pred.shape} does not match target {target.shape}')

                loss = F.mse_loss(pred.float(), target.float(), reduction='mean')
                
                lr = lr_scheduler.get_last_lr()[0]
                step_cnt += 1

                loss_dict = {
                    'step/step_cnt': step_cnt,
                    'step/epoch': epoch,
                    'lr/learning_rate': lr,
                    'train/denoise_loss': loss.detach().item(),
                    'train/total_loss': loss.detach().item(),
                }
                if accelerator.is_main_process:
                    for name, value in loss_dict.items():
                        if args.report_to == 'wandb':
                            wandb.log({name: value}, step=step_cnt)
                        elif args.report_to == 'tensorboard':
                            tb_writer.add_scalar(
                                name, value, global_step=step_cnt)

                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)).mean()
                train_loss += (
                    avg_loss.item() / args.gradient_accumulation_steps)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {'train_loss': train_loss}, step=global_step)
                train_loss = 0.0

                if (global_step % args.checkpointing_steps == 0
                        or global_step == args.max_train_steps):
                    next_epoch, next_batch = epoch, batch_index + 1
                    if next_batch == len(train_dataloader):
                        next_epoch, next_batch = epoch + 1, 0
                    checkpoint(next_epoch, next_batch)

            logs = {
                'loss': loss.detach().item(),
                'lr': lr_scheduler.get_last_lr()[0],
                'epoch': epoch,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        epoch += 1
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if tb_writer is not None:
        tb_writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bev_config', required=True)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--output_dir', default='results/BEVDiffuser_seg_dino')
    parser.add_argument(
        '--logging_dir',
        type=str,
        default='logs',
        help=(
            '[TensorBoard](https://www.tensorflow.org/tensorboard) log '
            'directory. Will default to *output_dir/runs/'
            '**CURRENT_DATETIME_HOSTNAME***.'
        ),
    )
    parser.add_argument(
        '--pretrained_model_name_or_path', default='stabilityai/stable-diffusion-2-1'
    )
    parser.add_argument('--pretrained_unet_checkpoint')
    parser.add_argument('--resume_from_checkpoint')
    parser.add_argument(
        '--prediction_type', choices=['sample', 'epsilon', 'v_prediction'], default='sample'
    )
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--max_train_steps', type=int, default=50000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', default='constant')
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--uncond_prob', type=float, default=0.2)
    parser.add_argument('--checkpointing_steps', type=int, default=10000)
    parser.add_argument('--checkpoints_total_limit', type=int, default=3)
    parser.add_argument(
        '--validation_batches', type=int, default=8, help='0 disables reconstruction validation'
    )
    parser.add_argument('--validation_timestep', type=int, default=100)
    parser.add_argument(
        '--report_to',
        type=str,
        default=None,
        choices=[None, 'wandb', 'tensorboard'],
        help=(
            'The integration used to report logs. Supported platforms are '
            '"tensorboard" and "wandb".'
        ),
    )
    parser.add_argument('--tracker_project_name', type=str, default=None)
    parser.add_argument('--tracker_run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument("--task_loss_scale", type=float, default=0.0)
    args = parser.parse_args()
    if (
        min(
            args.max_train_steps,
            args.train_batch_size,
            args.gradient_accumulation_steps,
            args.checkpointing_steps,
            args.checkpoints_total_limit,
        )
        < 1
    ):
        parser.error('Steps, batch size, accumulation and checkpoint limit must be positive')
    if not 0 <= args.uncond_prob <= 1 or args.validation_batches < 0:
        parser.error('Invalid conditioning probability or validation batch count')
    if args.pretrained_unet_checkpoint and args.resume_from_checkpoint:
        parser.error('Choose either pretrained_unet_checkpoint or resume_from_checkpoint')
    return args


if __name__ == '__main__':
    train()
