#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPUS=8
PORT=${PORT:-28516}

CONFIG="./projects/configs/diff_bevformer/layout_base_occ.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_base_original_bs4/checkpoint-50000"
MODEL_TARGET_CHECKPOINT="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_base_original_bs4/checkpoint-50000/bev_model.pth"
LOAD_FROM="./ckpts/r101_dcn_fcos3d_pretrain.pth"

# RESUME_FROM="../results/diff_bevformer/DiffBEVFormer_layout_base_occ/epoch_11.pth"

RUN_NAME="DiffBEVFormer_layout_base_occ_det_target_output_aligned"
WORK_DIR="../results/diff_bevformer/occ/baseline"
GRADIENT_MONITOR_ENABLED=True
GRADIENT_MONITOR_INTERVAL=200

if [[ ! -f "$UNET_CHECKPOINT_DIR/unet/diffusion_pytorch_model.safetensors" ]]; then
    echo "Stage 1 UNet checkpoint not found: $UNET_CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ ! -f "$MODEL_TARGET_CHECKPOINT" ]]; then
    echo "Detection target checkpoint not found: $MODEL_TARGET_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$LOAD_FROM" ]]; then
    echo "Student initialization checkpoint not found: $LOAD_FROM" >&2
    exit 1
fi

# NOTE: `VAR=... \` must be followed by the command itself. A comment line in
# between swallows the continuation, so PYTHONPATH would never reach torchrun.
export PYTHONPATH="$(dirname "$0")/..${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_layout_occ.py $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer-Occ3D' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --model-target-checkpoint=$MODEL_TARGET_CHECKPOINT \
    --load_from=$LOAD_FROM \
    --gradient-monitor-enabled=$GRADIENT_MONITOR_ENABLED \
    --gradient-monitor-interval=$GRADIENT_MONITOR_INTERVAL
