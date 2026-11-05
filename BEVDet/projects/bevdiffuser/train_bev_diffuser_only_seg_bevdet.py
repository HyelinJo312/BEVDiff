# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Following code is adapted from
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

BEVDet4D port of train_bev_diffuser_only_seg.py (semantic-only Stage 1 teacher), env jhl-bevdet.
Only the BEV-model side differs from the flagship script; the diffusion loop (noise, x0 MSE,
batch-level CFG dropout, AdamW, constant lr, clipping, accelerate checkpoints) is unchanged:
  - clean target x0 = frozen BEVDet4D.extract_img_feat(img_inputs, img_metas)[0][0] [B,256,128,128]
    (strict-loaded EMA weights, eval(), requires_grad_(False), no DDP wrapper);
  - data = mmdet3d (BEVDet) NuScenesDataset + LoadSemanticBEVPrior; no BEVFormer plugin imports
    (they fail to register under BEVDet's mmdet3d 1.0);
  - task loss and the in-training BEVFormer detection eval are removed (task_loss_scale must be 0);
  - TF32 is off unless --allow_tf32 (the flagship ran in fp32 with cuDNN disabled);
  - --pretrained_model_name_or_path accepts a local scheduler dir (HF SD-2.1 returns 401).
'''

import argparse
import copy
import datetime
import hashlib
import json
import logging
import math
import os, sys
import shutil
import subprocess
import random
import itertools
import wandb
import warnings
import accelerate
import datasets
import numpy as np
import torch

# cuDNN on for speed; TF32 is set from --allow_tf32 in train() (default off = fp32 numerics).
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers
import importlib

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import EMAModel

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, save_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from mmdet.apis import set_random_seed
from model_utils import build_unet_v2
from torch.utils.tensorboard import SummaryWriter


logger = get_logger(__name__, log_level="INFO")

BEVDET_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def build_frozen_bevdet(bev_cfg, checkpoint):
    """Frozen BEV target: BEVDet(4D) from bev_cfg.model with strictly loaded weights."""
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'--bev_checkpoint not found: {checkpoint!r}')
    model = build_model(copy.deepcopy(bev_cfg.model))
    load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    if getattr(model, 'align_after_view_transfromation', False):
        raise ValueError('expected align_after_view_transfromation=False (the C0 training path)')
    if hasattr(model, 'with_prev'):
        model.with_prev = True  # SequentialControlHook is not used here; keep the temporal path on
    model.requires_grad_(False)
    model.eval()
    return model


def state_fingerprint(module):
    """Cheap checksum of all parameters and buffers (detects any change of the frozen target)."""
    with torch.no_grad():
        return float(sum(t.double().sum() for t in itertools.chain(module.parameters(), module.buffers())
                         if t.is_floating_point()))


def _md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _git(*cmd):
    try:
        return subprocess.run(['git', '-C', BEVDET_ROOT, *cmd], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f'unavailable: {e}'


def write_run_info(args, bev_cfg, prior_cfg, noise_scheduler, unet, num_examples, global_step):
    """Provenance of a Stage 1 run: args, frozen-target checkpoint, cache contract, code state."""
    from projects.bevdiffuser.semantic_bev_cache import read_cache_manifest
    info = dict(
        time=datetime.datetime.now().isoformat(timespec='seconds'),
        start_global_step=global_step,
        argv=sys.argv,
        args={k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
              for k, v in vars(args).items()},
        bev_checkpoint=dict(path=os.path.abspath(args.bev_checkpoint),
                            bytes=os.path.getsize(args.bev_checkpoint), md5=_md5(args.bev_checkpoint)),
        semantic_prior=dict(prior_cfg, cache_root_abs=os.path.abspath(prior_cfg['cache_root']),
                            manifest_contract=read_cache_manifest(prior_cfg['cache_root'])),
        noise_scheduler=dict(noise_scheduler.config),
        unet_params=sum(p.numel() for p in unet.parameters()),
        train_examples=num_examples,
        torch=torch.__version__, cuda=torch.version.cuda,
        tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32),
        bevdet_git=dict(head=_git('rev-parse', 'HEAD'), status=_git('status', '--short')),
    )
    name = 'stage1_run_info.json' if global_step == 0 else f'stage1_run_info_resume_{global_step}.json'
    with open(os.path.join(args.output_dir, name), 'w') as f:
        json.dump(info, f, indent=2, default=str)
    bev_cfg.dump(os.path.join(args.output_dir, 'bev_config_dump.py'))

def train():
    args = parse_args()

    resume_output_dir = None
    if args.resume_from_checkpoint:
        args.output_dir = os.path.dirname(os.path.normpath(args.resume_from_checkpoint))
        resume_output_dir = args.output_dir

    bev_cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        bev_cfg.merge_from_dict(args.cfg_options)
    if args.task_loss_scale != 0:
        raise ValueError('BEVDet Stage 1 is semantic-only: --task_loss_scale must be 0')
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if resume_output_dir is not None:
        logger.info(f"resume_from_checkpoint detected, reusing output dir {resume_output_dir}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    # Frozen clean-target model (BEVDet4D C0). Not wrapped in DDP: it has no trainable params.
    bev_model = build_frozen_bevdet(bev_cfg, args.bev_checkpoint)
    bev_fingerprint = state_fingerprint(bev_model)
    prior_cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                 dict(next(t for t in bev_cfg.data.train.pipeline if t['type'] == 'LoadSemanticBEVPrior')).items()}
    expected_latent = (bev_cfg._dim_, bev_cfg.bev_h_, bev_cfg.bev_w_)
    expected_prior = (prior_cfg['cache_shape'][0], bev_cfg.prior_h_, bev_cfg.prior_w_)

    unet = build_unet_v2(bev_cfg.unet)
    if args.pretrained_unet_checkpoint is not None and (os.path.isfile(args.pretrained_unet_checkpoint) or os.path.isdir(args.pretrained_unet_checkpoint)):
        unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder="unet")
        # train only the downsample and upsample layers
        unet.requires_grad_(False)
        unet.downsample_blocks.requires_grad_(True)
        unet.upsample_blocks.requires_grad_(True)

    num_fdn = sum(1 for m in unet.modules() if type(m).__name__ == 'SBAMResBlock')
    assert version.parse(accelerate.__version__) >= version.parse("0.16.0"), "accelerate 0.16.0 or above is required"

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        for model in models:
            # save_state runs on every rank; only one rank writes shared weights.
            if accelerator.is_main_process:
                model.save_pretrained(os.path.join(output_dir, "unet"))

            # Every rank must pop the weight to disable the default model save.
            weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            model.from_pretrained(os.path.join(input_dir, "unet"))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW

    trained_params = list(unet.parameters())
    if args.task_loss_scale != 0:
        trained_params += list(bev_model.parameters())
    
    learning_rate = args.learning_rate

    optimizer = optimizer_cls(
        trained_params,
        lr=learning_rate,
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

    # DataLoaders creation (mmdet builder: DistributedGroupSampler when dist, shuffled per epoch).
    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=args.train_batch_size,
        workers_per_gpu=args.dataloader_num_workers,
        num_gpus=get_dist_info()[1],
        dist=(args.launcher != 'none'),
        seed=args.seed,
        shuffle=True,
    )

    def get_segmaps_cond(segmaps):
        if np.random.rand() < args.uncond_prob:
            cond = torch.zeros_like(segmaps)
        else:
            cond = segmaps
        return cond
    
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype
    bev_model.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    tb_writer = None
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:   
        tracker_config = dict(vars(args))
        if args.report_to == 'tensorboard' and isinstance(tracker_config.get('cfg_options'), dict):
            tracker_config['cfg_options'] = json.dumps(tracker_config['cfg_options'], sort_keys=True)

        if args.resume_from_checkpoint:
            resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
            args.tracker_run_name = f"{args.tracker_run_name}-resume-{resume_ckpt_number}"

        init_kwargs = {}
        if args.report_to == "wandb":
            wandb.init(project=args.tracker_project_name,
                       name=args.tracker_run_name,
                       id=args.tracker_run_name)
            init_kwargs = {
                "wandb" : {
                    "name" : args.tracker_run_name
                }
            }

        accelerator.init_trackers(project_name=args.tracker_project_name, 
                                  config=tracker_config,
                                  init_kwargs=init_kwargs)
        
        if args.report_to == "tensorboard":
            tb_logdir = os.path.join(args.output_dir, "tensorboard_logs")
            tb_writer = SummaryWriter(log_dir=tb_logdir)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    is_training_sd21 = args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-1"

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Is SD21: {is_training_sd21}")
    logger.info(f"  Frozen BEV target: {bev_cfg.model.type} <- {args.bev_checkpoint} "
                f"(latents {expected_latent}, target_scale {args.target_scale})")
    logger.info(f"  Semantic BEV condition source: NumPy cache {prior_cfg['cache_root']} "
                f"{tuple(prior_cfg['cache_shape'])} -> warped prior {expected_prior}")
    logger.info(f"  TF32: {args.allow_tf32}")
    logger.info(f'[UNet] SBAMResBlock count: {num_fdn}')

    global_step = 0
    first_epoch = 0
    step_cnt = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        random_state_src = os.path.join(resume_path, "random_states_0.pkl")

        # Older checkpoints were saved only on rank 0, so only one random state
        # file may exist. Backfill missing rank files before loading.
        if accelerator.is_main_process and os.path.isfile(random_state_src):
            for process_idx in range(accelerator.num_processes):
                random_state_dst = os.path.join(resume_path, f"random_states_{process_idx}.pkl")
                if not os.path.exists(random_state_dst):
                    shutil.copyfile(random_state_src, random_state_dst)
                    logger.info(f"Backfilled missing random state file: {random_state_dst}")
        accelerator.wait_for_everyone()

        logger.info(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(resume_path)
        bev_resume_path = os.path.join(resume_path, "bev_model.pth")
        if os.path.isfile(bev_resume_path):
            load_checkpoint(bev_model, bev_resume_path, map_location="cpu")
            logger.info(f"Loaded BEV model state from {bev_resume_path}")
        global_step = int(resume_path.split("-")[-1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        # resume step indicates how many data we should skip in this epoch

        # change step_cnt
        step_cnt = global_step * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        write_run_info(args, bev_cfg, prior_cfg, noise_scheduler, unet, len(train_dataset), global_step)
    grads_checked = False

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        # This MMCV dataloader is not wrapped by accelerator.prepare().
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        # batch: ['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'layout_obj_classes', 'layout_obj_bboxes', 'layout_obj_is_valid', 'layout_obj_names', 'default_obj_names', 'seg_maps']
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # For Resume from checkpoint, Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % 10 == 0:
                    logger.info(f"skipping data {step} / {resume_step}")
                continue

            with accelerator.accumulate(unet):
                # Get BEV
                img_inputs = [t.to(accelerator.device, non_blocking=True) for t in batch['img_inputs']]
                img_metas = batch['img_metas'].data[0]
                with torch.no_grad():
                    latents = bev_model.extract_img_feat(img_inputs, img_metas)[0][0].detach()
                latents = latents.float().contiguous() * args.target_scale
                if tuple(latents.shape[1:]) != expected_latent:
                    raise RuntimeError(f'BEV latents {tuple(latents.shape)} != [B, {expected_latent}]')

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                max_timestep = noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, max_timestep, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "sample":
                    target = latents
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                depth_maps = None
                semantic_prob = torch.stack(batch['semantic_bev_probabilities'].data[0], dim=0).to(latents.device)
                if tuple(semantic_prob.shape[1:]) != expected_prior:
                    raise RuntimeError(f'semantic prior {tuple(semantic_prob.shape)} != [B, {expected_prior}]')
                cond = get_segmaps_cond(semantic_prob)
                uncond_batch = cond is not semantic_prob
                cond_kwargs = {'semantic_prob': cond}
                seg_cond = None

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, img_metas, seg_cond, depth_maps=depth_maps, **cond_kwargs)

                denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                task_loss = 0  # semantic-only: task_loss_scale == 0 is enforced at startup

                total_loss = denoise_loss + args.task_loss_scale * task_loss
                
                # get learing rate
                lr = lr_scheduler.get_last_lr()[0]

                step_cnt += 1

                loss_dict = {
                    "step/step_cnt" : step_cnt,
                    "step/epoch": epoch,
                    "lr/learning_rate" : lr,
                    "train/denoise_loss": denoise_loss,
                    "train/task_loss": task_loss,
                    "train/total_loss": total_loss,
                    "train/uncond_batch": float(uncond_batch),
                }

                if accelerator.is_main_process:
                    for name, value in loss_dict.items():
                        if args.report_to == "wandb":
                            wandb.log({name : value}, step=step_cnt) 
                        elif args.report_to == "tensorboard":
                            tb_writer.add_scalar(name, value, global_step=step_cnt)

                loss = total_loss 

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                                
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients and not grads_checked:
                    no_grad = [n for n, p in unet.named_parameters() if p.requires_grad and p.grad is None]
                    if no_grad:
                        logger.warning(f'{len(no_grad)} trainable UNet params got no grad, e.g. {no_grad[:5]}')
                    else:
                        logger.info('All trainable UNet params received gradients on the first step')
                    grads_checked = True
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save checkpoint 
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        # The frozen target is not saved per checkpoint: it is the --bev_checkpoint
                        # file recorded (path + md5) in stage1_run_info.json.
                        logger.info(f"Saved state to {save_path}")
                    if bev_model.training or state_fingerprint(bev_model) != bev_fingerprint:
                        raise RuntimeError('frozen BEVDet target changed during Stage 1')

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch":epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()
    if tb_writer is not None:
        tb_writer.close()   
    
    
def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument('--bev_config', 
                        default="",
                        help='test config file path')
    
    parser.add_argument('--bev_checkpoint', 
                        default="",
                        help='checkpoint file')
    
    parser.add_argument('--pretrained_unet_checkpoint', 
                        default=None,
                        help='checkpoint file')
    
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="HF id or local dir with scheduler/scheduler_config.json (only the scheduler is used); "
             "e.g. projects/bevdiffuser/sd21_scheduler, a local copy of the SD-2.1 scheduler.",
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 matmul/cuDNN. Off by default: the flagship Stage 1 ran in fp32.",
    )

    parser.add_argument(
        "--target_scale",
        type=float,
        default=1.0,
        help="Multiply the frozen BEV target x0 by this factor (1.0 = raw features, as in the flagship). "
             "Stage 2 must use the same factor.",
    )
    
    parser.add_argument(
        "--uncond_prob", 
        default=0.2, 
        type=float, 
        help="The probability of replacing caption with empty string."
    )
    
    parser.add_argument(
        "--uncond_prob_seg", 
        default=0.2, 
        type=float, 
        help="The probability of replacing caption with empty string."
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default='results/test',
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )


    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=24500,
        # required=True,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        # required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'sample' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=[None, "wandb", "tensorboard"],
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        # required=True,
        default=8000,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        # required=True,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None
    )

    # # below are additional params
    parser.add_argument(
        "--task_loss_scale", 
        type=float, 
        default=0.0
    )
    
    parser.add_argument(
        "--task_loss_switch_step",
        type=int,
        default=-1,
        help="If >0, after this many optimization steps use multi_feat for task loss (if available). Default 0 -> always use model_pred."
    )
    

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    return args

if __name__ == "__main__":
    train()