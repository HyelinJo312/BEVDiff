#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no

# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

GPUS=8
PORT=${PORT:-28506}

CONFIG="./projects/configs/diff_bevformer/layout_base.py"
# CONFIG="./projects/configs/diff_bevformer/layout_base.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_base_original_bs4/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_r101_dcn_24ep.pth" # BEVFormer-base model
# LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"

# RESUME_FROM="../results/diff_bevformer/DiffBEVFormer_tiny_original/DiffBEVFormer_tiny_original/epoch_11.pth"

# RUN_NAME="DiffBEVFormer_base_original"
RUN_NAME="DiffBEVFormer_base_original"
WORK_DIR="../results/diff_bevformer/baseline"

# export PYTHONWARNINGS="ignore"

# NOTE: `VAR=... \` must be followed by the command itself. A comment line in
# between swallows the continuation, so PYTHONPATH would never reach torchrun.
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
