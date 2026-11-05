# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Following code is adapted from 
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''

import argparse
import atexit
import hashlib
import json
import logging
import math
import os, sys
import random
import itertools
import wandb
import warnings
import accelerate
import datasets
import numpy as np
import torch

# torch.backends.cudnn.enabled = False
# cuDNN 및 convolution 알고리즘 자동 선택
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# FP32 convolution과 행렬 연산에 TF32 사용 허용
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

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
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file as load_safetensors
# from diffusers.training_utils import EMAModel

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.apis import set_random_seed
from model_utils import build_unet_v2, get_bev_model_vidar
from BEVFormer.projects.bevdiffuser.analyze.vidar_bev_target import EXPECTED_SHAPE, EXPECTED_RANGE
from projects.bevdiffuser.data_utils import ViDARSemanticCacheDataset  # register token/cache dataset
from torch.utils.tensorboard import SummaryWriter


logger = get_logger(__name__, log_level="INFO")

def train():
    args = parse_args()
    if args.task_loss_scale != 0:
        raise ValueError('ViDAR Stage 1 is head-free: --task_loss_scale must be 0')
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = os.path.abspath(args.resume_from_checkpoint)
        checkpoint_name = os.path.basename(resume_path)
        if (not os.path.isdir(resume_path) or not checkpoint_name.startswith('checkpoint-')
                or not checkpoint_name.split('checkpoint-', 1)[1].isdigit()):
            raise ValueError('Resume requires a ViDAR checkpoint-N directory')
        args.output_dir = os.path.dirname(resume_path)

    bev_cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        if any(key.split('.', 1)[0] == 'model' for key in args.cfg_options):
            raise ValueError('Change the ViDAR model in --bev_config; the worker reads that file')
        bev_cfg.merge_from_dict(args.cfg_options)
    args.bev_checkpoint = os.path.abspath(os.path.expanduser(args.bev_checkpoint)) if args.bev_checkpoint else ''
    if not args.bev_checkpoint or not os.path.isfile(args.bev_checkpoint):
        raise FileNotFoundError('Pass the ViDAR checkpoint with --bev_checkpoint')

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
    if noise_scheduler.config.prediction_type != 'sample':
        raise ValueError('ViDAR Stage 1 uses x0/sample prediction only')
    
    unet = build_unet_v2(bev_cfg.unet)

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
            weights_path = os.path.join(input_dir, 'unet', SAFETENSORS_WEIGHTS_NAME)
            model.load_state_dict(load_safetensors(weights_path, device='cpu'), strict=True)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW

    trained_params = list(unet.parameters())
    
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
        train_dataset = build_dataset(bev_cfg.data.train, 
                                      default_args={
                                          'pc_range': bev_cfg.point_cloud_range,
                                          'use_3d_bbox': bev_cfg.use_3d_bbox,
                                          'num_classes': bev_cfg.num_classes,
                                          'num_bboxes': bev_cfg.num_bboxes,
                                      })
        
      
    # DataLoaders creation:
    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=args.train_batch_size, 
        workers_per_gpu=args.dataloader_num_workers,
        num_gpus=get_dist_info()[1],
        dist=(args.launcher != 'none'),
        seed=args.seed,
        shuffler_sampler=bev_cfg.data.shuffler_sampler,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
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

    bev_model = get_bev_model_vidar(args, split='train', device=accelerator.device)
    atexit.register(bev_model.close)
    
    run_provenance = dict(bev_model.provenance)
    run_provenance['effective_stage1_config_sha256'] = hashlib.sha256(
        bev_cfg.pretty_text.encode('utf-8')).hexdigest()
    run_provenance['training_setup'] = dict(
        scheduler=args.pretrained_model_name_or_path, prediction_type=args.prediction_type,
        uncond_prob=args.uncond_prob, seed=args.seed,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps)
    logger.info('ViDAR Stage 1 provenance: %s', json.dumps(run_provenance, sort_keys=True))
    
    if resume_path is not None:
        manifest = os.path.join(resume_path, 'vidar_target_source.json')
        with open(manifest) as stream:
            saved_provenance = json.load(stream)
        if saved_provenance != run_provenance:
            raise ValueError('Resume Stage 1 provenance differs from current source or setup')

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
    if hasattr(train_dataset, 'use_semantic_bev_cache'):
        condition_source = (f'NumPy cache: {train_dataset.semantic_bev_cache_root}'
                            if train_dataset.use_semantic_bev_cache else 'live SAM3/Metric3D')
        logger.info(f"  Semantic BEV condition source: {condition_source}")
    logger.info(f'[UNet] SBAMResBlock count: {num_fdn}')

    global_step = 0
    first_epoch = 0
    step_cnt = 0

    if resume_path is not None:
        accelerator.load_state(resume_path)
        global_step = int(os.path.basename(resume_path).split('-')[-1])
        if global_step >= args.max_train_steps:
            raise ValueError('Resume checkpoint is at or beyond max_train_steps')
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = (global_step * args.gradient_accumulation_steps) % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps)
        step_cnt = global_step * args.gradient_accumulation_steps

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
            if resume_path is not None and epoch == first_epoch and step < resume_step:
                continue
            with accelerator.accumulate(unet):
                # The worker reads deterministic full-resolution ViDAR images by
                # the current frame token. Stage 1's 50x50 semantic cache is
                # attached to the same current frame by the training dataset.
                if 'sample_token' in batch:
                    tokens = batch['sample_token'].data[0]
                    img_metas = None
                    if 'img' in batch or 'depth_maps' in batch:
                        raise ValueError('Token-only ViDAR batch unexpectedly contains images or depth')
                else:
                    imgs = batch['img'].data[0]
                    len_queue = imgs.size(1)
                    img_metas = [each[len_queue - 1] for each in batch['img_metas'].data[0]]
                    tokens = [meta['sample_idx'] for meta in img_metas]
                
                latents = bev_model.get(tokens, accelerator.device)
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
                if 'depth_maps' in batch.keys():
                    depth_maps = torch.stack(batch['depth_maps'].data[0], dim=0)

                cond_kwargs = {}
                if 'semantic_bev_probabilities' not in batch:
                    raise ValueError('ViDAR Stage 1 requires cached semantic BEV probabilities')
                semantic_prob = torch.stack(batch['semantic_bev_probabilities'].data[0], dim=0).to(latents.device)
                cond_kwargs['semantic_prob'] = get_segmaps_cond(semantic_prob)
                seg_cond = None

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, img_metas, seg_cond, depth_maps=depth_maps, **cond_kwargs)[0]

                denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                task_loss = 0
                total_loss = denoise_loss
                
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
                    if os.path.exists(save_path):
                        raise FileExistsError(f'Refusing to overwrite checkpoint: {save_path}')
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        with open(os.path.join(save_path, 'vidar_target_source.json'), 'x') as stream:
                            json.dump(run_provenance, stream, indent=2, sort_keys=True)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch":epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()
    bev_model.close()
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
        choices=[
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1"
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        default='sample',
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
        default=None,
        help=("Leave unset; the ViDAR variant never prunes checkpoints."),
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
