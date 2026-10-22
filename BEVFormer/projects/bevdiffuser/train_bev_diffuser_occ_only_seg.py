# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Stage-1 semantic-only BEV diffusion training for Occ3D.

This is the occupancy counterpart of ``train_bev_diffuser_only_seg.py``.
The clean target is produced by a frozen BEVFormerOcc encoder. The diffusion
UNet is conditioned only on dense semantic BEV priors, from cached Metric3D semantic probabilities. Evaluation is handled separately by
``test_bev_diffuser_occ_only_seg.py``.
"""

import argparse
import logging
import math
import os
import random
import shutil
import sys
import warnings
import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb

# torch.backends.cudnn.enabled = False
# cuDNN 및 convolution 알고리즘 자동 선택
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# FP32 convolution과 행렬 연산에 TF32 사용 허용
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.bevdiffuser.model_utils import build_unet_v2
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler
from projects.bevdiffuser import data_utils  # Register the occupancy/cache dataset.
from projects.mmdet3d_plugin.bevformer.detectors.bevformer_occ import BEVFormerOcc
from projects.bevdiffuser.occ_diffusion_utils import (
    get_occ_model, occ_task_modules, occ_task_parameters,
    set_occ_task_train, load_diffusion_weights)
from projects.bevdiffuser.test_bev_diffuser_occ_only_seg import evaluate_occ

logger = get_logger(__name__, log_level="INFO")


def _restore_resume_random_state(resume_path, process_index):
    """Restore this process's RNG from a trusted local Accelerate checkpoint."""
    rng_path = os.path.join(resume_path, f"random_states_{process_index}.pkl")
    if not os.path.isfile(rng_path):
        raise FileNotFoundError(f"Missing RNG state for rank {process_index}: {rng_path}")
    try:
        # Accelerate 0.20.3 calls torch.load without weights_only=False, which
        # cannot read these RNG files under newer PyTorch defaults.
        states = torch.load(rng_path, map_location="cpu", weights_only=False)
        required = ("random_state", "numpy_random_seed", "torch_manual_seed",
                    "torch_cuda_manual_seed")
        if not isinstance(states, dict) or any(key not in states for key in required):
            raise ValueError(f"Expected RNG state keys: {required}")
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    except Exception as exc:
        raise RuntimeError(f"Failed to restore RNG state for rank {process_index} from {rng_path}") from exc
    return rng_path


def _bev_tokens_to_map(bev, bev_cfg):
    """Convert BEVFormer tokens to diffusion layout ``[B,C,Y,X]``."""
    if bev.dim() == 4:
        expected = (bev_cfg._dim_, bev_cfg.bev_h_, bev_cfg.bev_w_)
        if tuple(bev.shape[1:]) != expected:
            raise ValueError(f"Unexpected BEV map shape {tuple(bev.shape)}; expected "
                             f"[B,{expected[0]},{expected[1]},{expected[2]}]")
        return bev.contiguous()

    if bev.dim() != 3:
        raise ValueError(f"BEV feature must be 3D or 4D, got {bev.shape}")

    h, w, channels = bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_
    if bev.shape[1:] == (h * w, channels):
        tokens = bev
    elif bev.shape[0] == h * w and bev.shape[2] == channels:
        tokens = bev.permute(1, 0, 2)
    else:
        raise ValueError(f"Cannot infer BEV token axis from {tuple(bev.shape)} for "
                         f"H={h}, W={w}, C={channels}")
    return tokens.reshape(tokens.shape[0], h, w, channels).permute(0, 3, 1, 2).contiguous()


def _bev_map_to_tokens(bev_map):
    """Convert diffusion layout ``[B,C,Y,X]`` to ``[B,Y*X,C]``."""
    return bev_map.permute(0, 2, 3, 1).reshape(bev_map.shape[0], -1, bev_map.shape[1]).contiguous()


def _container_data(value):
    """Extract one-rank data from an MMCV container/test-aug wrapper."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Only single-scale, non-flipped batches are supported")
        value = value[0]
    if isinstance(value, DataContainer):
        value = value.data[0]
    return value


def _batch_tensor(batch, key, device):
    value = _container_data(batch[key])
    if isinstance(value, (list, tuple)):
        value = torch.stack(value, dim=0)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must contain tensors, got {type(value)}")
    return value.to(device=device, non_blocking=True)


def _current_img_metas(batch):
    """Extract current-frame metadata from train queues or test wrappers."""
    metas = _container_data(batch["img_metas"])
    current = []
    for sample_metas in metas:
        if isinstance(sample_metas, dict) and sample_metas and all(isinstance(key, int) for key in sample_metas):
            sample_metas = sample_metas[max(sample_metas)]
        current.append(sample_metas)
    return current


def train():
    args = parse_args()

    resume_output_dir = None
    if args.resume_from_checkpoint:
        args.output_dir = os.path.dirname(os.path.normpath(args.resume_from_checkpoint))
        resume_output_dir = args.output_dir

    bev_cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        bev_cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(bev_cfg)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if resume_output_dir is not None:
        logger.info(f"resume_from_checkpoint detected, reusing output dir "
                    f"{resume_output_dir}")

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

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_scheduler = DDIMGuidedScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
        ddim_scheduler.register_to_config(prediction_type=args.prediction_type)

    bev_model = get_occ_model(args, bev_cfg)
    bev_model.requires_grad_(False)
    occ_model = bev_model.module
    if not isinstance(occ_model, BEVFormerOcc):
        raise TypeError('Use the native occupancy Stage 1 config')
    if args.task_loss_scale != 0:
        for module in occ_task_modules(occ_model):
            module.requires_grad_(True)

    unet = build_unet_v2(bev_cfg.unet)
    if args.pretrained_unet_checkpoint is not None and (os.path.isfile(args.pretrained_unet_checkpoint) or os.path.isdir(args.pretrained_unet_checkpoint)):
        unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder="unet")
        unet.requires_grad_(True)
        # unet.downsample_blocks.requires_grad_(True)
        # unet.upsample_blocks.requires_grad_(True)

    def get_occ_task_loss(bev_map, batch):
        device = bev_map.device
        losses = bev_model(
            return_loss=True,
            given_bev=_bev_map_to_tokens(bev_map),
            voxel_semantics=_batch_tensor(batch, "voxel_semantics", device),
            mask_camera=_batch_tensor(batch, "mask_camera", device),
            mask_lidar=_batch_tensor(batch, "mask_lidar", device),
            img_metas=_current_img_metas(batch),
        )
        loss, _ = occ_model._parse_losses(losses)
        return loss

    num_sbam = sum(1 for module in unet.modules() if type(module).__name__ == "SBAMResBlock")
    assert version.parse(accelerate.__version__) >= version.parse("0.16.0"), "accelerate 0.16.0 or above is required"

    def save_model_hook(models, weights, output_dir):
        for model in models:
            if accelerator.is_main_process:
                model.save_pretrained(os.path.join(output_dir, "unet"))
            weights.pop()

    def load_model_hook(models, input_dir):
        while models:
            model = models.pop()
            load_diffusion_weights(model, os.path.join(input_dir, "unet"))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trained_params = list(unet.parameters())
    if args.task_loss_scale != 0:
        trained_params += list(occ_task_parameters(occ_model))

    optimizer = torch.optim.AdamW(
        trained_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    with accelerator.main_process_first():
        train_dataset = build_dataset(bev_cfg.data.train)
        val_dataset = build_dataset(bev_cfg.data.test, default_args=dict(test_mode=True))

    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=args.train_batch_size,
        workers_per_gpu=args.dataloader_num_workers,
        num_gpus=get_dist_info()[1],
        dist=(args.launcher != "none"),
        seed=args.seed,
        shuffler_sampler=bev_cfg.data.shuffler_sampler,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=bev_cfg.data.get("val", {}).get("samples_per_gpu", 1),
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != "none"),
        shuffle=False,
        seed=args.seed,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    def get_segmaps_cond(x):
        if np.random.rand() < args.uncond_prob:
            return torch.zeros_like(x)
        return x

    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    bev_model.to(accelerator.device, dtype=torch.float32)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    tb_writer = None
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        if args.resume_from_checkpoint:
            checkpoint_number = args.resume_from_checkpoint.split("-")[-1]
            args.tracker_run_name = f"{args.tracker_run_name}-resume-{checkpoint_number}"

        init_kwargs = {}
        if args.report_to == "wandb":
            wandb.init(project=args.tracker_project_name, name=args.tracker_run_name, id=args.tracker_run_name)
            init_kwargs = {"wandb": {"name": args.tracker_run_name}}

        accelerator.init_trackers(project_name=args.tracker_project_name, config=tracker_config,
                                  init_kwargs=init_kwargs)
        if args.report_to == "tensorboard":
            tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard_logs"))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running Occ3D Stage-1 diffusion training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  SBAMResBlock count = {num_sbam}")
    logger.info(f"  Auxiliary occupancy loss scale = {args.task_loss_scale}")

    global_step = 0
    first_epoch = 0
    step_count = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(resume_path)
        rng_path = _restore_resume_random_state(resume_path, accelerator.process_index)
        logger.info(f"Restored RNG state for rank {accelerator.process_index} from {rng_path}", main_process_only=False)
        occ_resume_path = os.path.join(resume_path, "occ_model.pth")
        if args.task_loss_scale != 0 and os.path.isfile(occ_resume_path):
            load_checkpoint(bev_model, occ_resume_path, map_location="cpu")
            logger.info(f"Loaded occupancy model from {occ_resume_path}")

        global_step = int(resume_path.split("-")[-1])
        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        step_count = resume_global_step

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % 10 == 0:
                    logger.info(f"Skipping batch {step} / {resume_step}")
                continue

            with accelerator.accumulate(unet):
                with torch.no_grad():
                    bev = bev_model(return_loss=False, only_bev=True, **batch).detach()
                latents = _bev_tokens_to_map(bev, bev_cfg)
                # BEV history extraction changes the model's train/eval state.
                # BN3d follows baseline training only when the task is trained.
                set_occ_task_train(occ_model, args.task_loss_scale != 0)

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size, ),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prediction_type = noise_scheduler.config.prediction_type
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "sample":
                    target = latents
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {prediction_type}")

                img_metas = _current_img_metas(batch)

                condition_kwargs = {}
                if 'semantic_bev_probabilities' in batch:
                    semantic_prob = _batch_tensor(batch, 'semantic_bev_probabilities', accelerator.device)
                    condition_kwargs['semantic_prob'] = get_segmaps_cond(semantic_prob)
                    seg_cond = None
                    depth_maps = None
                else:
                    seg_maps = _batch_tensor(batch, 'seg_maps', accelerator.device)
                    seg_cond = get_segmaps_cond(seg_maps)
                    depth_maps = _batch_tensor(batch, 'depth_maps', accelerator.device) if 'depth_maps' in batch else None

                model_pred = unet(noisy_latents, timesteps, img_metas, seg_cond, depth_maps=depth_maps, **condition_kwargs)

                denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if args.task_loss_scale > 0 and prediction_type == "sample":
                    task_loss = get_occ_task_loss(model_pred, batch)
                else:
                    task_loss = denoise_loss.new_zeros(())

                total_loss = denoise_loss + args.task_loss_scale * task_loss

                learning_rate = lr_scheduler.get_last_lr()[0]
                step_count += 1
                loss_dict = {
                    "step/step_cnt": step_count,
                    "step/epoch": epoch,
                    "lr/learning_rate": learning_rate,
                    "train/denoise_loss": denoise_loss,
                    "train/occ_task_loss": task_loss,
                    "train/total_loss": total_loss,
                }
                if accelerator.is_main_process:
                    for name, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().item()
                        if args.report_to == "wandb":
                            wandb.log({name: value}, step=step_count)
                        elif args.report_to == "tensorboard":
                            tb_writer.add_scalar(name, value, global_step=step_count)

                avg_loss = accelerator.gather(total_loss.detach().repeat(batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trained_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                path for path in os.listdir(args.output_dir) if path.startswith("checkpoint-")
                            ]
                            checkpoints.sort(key=lambda path: int(path.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                remove_count = len(checkpoints) - args.checkpoints_total_limit + 1
                                for checkpoint in checkpoints[:remove_count]:
                                    shutil.rmtree(os.path.join(args.output_dir, checkpoint))

                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process and args.task_loss_scale != 0:
                        save_checkpoint(bev_model, filename=os.path.join(save_path, "occ_model.pth"))

                    if global_step % args.checkpointing_steps == 0 and global_step >= 30000:
                        logger.info(f"Evaluating Occ3D at epoch {epoch}, step {global_step}")
                        unet.eval()
                        set_occ_task_train(occ_model, False)
                        eval_results = evaluate_occ(
                            unet=accelerator.unwrap_model(unet),
                            bev_model=bev_model,
                            noise_scheduler=ddim_scheduler,
                            dataset=val_dataset,
                            dataloader=val_dataloader,
                            bev_cfg=bev_cfg,
                            output_dir=os.path.join(save_path, "val"),
                            noise_timesteps=300,
                            denoise_timesteps=300,
                            num_inference_steps=5,
                            guidance_scale=4,
                        )
                        if accelerator.is_main_process:
                            for metric, score in eval_results.items():
                                metric_name = f"val/{metric}"
                                if args.report_to == "wandb":
                                    wandb.log({metric_name: score}, step=global_step)
                                elif args.report_to == "tensorboard":
                                    tb_writer.add_scalar(metric_name, score, global_step=global_step)
                        unet.train()
                        set_occ_task_train(occ_model, args.task_loss_scale != 0)

            progress_bar.set_postfix(loss=total_loss.detach().item(), lr=lr_scheduler.get_last_lr()[0], epoch=epoch)
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    if tb_writer is not None:
        tb_writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train semantic-only Stage-1 BEV diffusion for Occ3D")
    parser.add_argument("--bev_config", required=True)
    parser.add_argument("--bev_checkpoint", required=True,
                        help="Frozen BEVFormerOcc epoch checkpoint")
    parser.add_argument("--pretrained_unet_checkpoint", default=None)

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--gpus", type=int)
    group_gpus.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="pytorch")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1",
                        choices=[
                            "CompVis/stable-diffusion-v1-4",
                            "stabilityai/stable-diffusion-2-1",
                        ])
    parser.add_argument("--uncond_prob", default=0.2, type=float)
    parser.add_argument("--output_dir", type=str, default="results/test")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--prediction_type", type=str, default="sample", choices=["epsilon", "sample", "v_prediction"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "wandb", "tensorboard"])
    parser.add_argument("--checkpointing_steps", type=int, default=8000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tracker_project_name", type=str, default="Occ3D-BEVDiffuser")
    parser.add_argument("--tracker_run_name", type=str, default="BEVDiffuser_occ_onlyseg")
    parser.add_argument("--task_loss_scale", type=float, default=0.0,
                        help="Optional auxiliary Occ3D loss; keep 0 for head-free Stage 1")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError("--options and --cfg-options cannot both be specified")
    if args.options:
        warnings.warn("--options is deprecated; use --cfg-options instead")
        args.cfg_options = args.options
    if not 0.0 <= args.uncond_prob <= 1.0:
        raise ValueError("--uncond_prob must be in [0, 1]")
    return args


if __name__ == "__main__":
    train()
