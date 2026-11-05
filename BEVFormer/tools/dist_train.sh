#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no

# stabilityai/stable-diffusion-2-1 was removed from the Hub (401); its tokenizer,
# text_encoder and scheduler are served from the local HF cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPUS=4
PORT=${PORT:-28506}

CONFIG="./projects/configs/diff_bevformer/layout_tiny.py"
# CONFIG="./projects/configs/diff_bevformer/layout_base.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_tiny_original_bs4/checkpoint-50000"
# LOAD_FROM="./ckpts/bevformer_r101_dcn_24ep.pth" # BEVFormer-base model
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"

# RESUME_FROM="../results/diff_bevformer/DiffBEVFormer_tiny_original/DiffBEVFormer_tiny_original/epoch_11.pth"

RUN_NAME="DiffBEVFormer_tiny_original_rerun_v2"
WORK_DIR="../results/diff_bevformer0/det/baseline"

# export PYTHONWARNINGS="ignore"

export PYTHONPATH="$(dirname "$0")/..${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \
