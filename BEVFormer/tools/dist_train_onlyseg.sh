#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_v2_da3.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/BEVDiffuser_tiny_onlyseg_sam3_v1/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_onlyseg_sam_mgd_v1_t100_alpha100_lambda_0.55"
WORK_DIR="../results/diff_bevformer"

# RESUME_FROM="../results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_da3_no-mgd/epoch_18.pth"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_onlyseg.py $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
