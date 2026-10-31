#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# export ACCELERATE_MIXED_PRECISION=no
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4
PORT=${PORT:-28503}
CONFIG="./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_metric3d_stubev.py"
# Original semantic teacher; the config enables its legacy semantic stem.
UNET_CHECKPOINT_DIR="../results/bevdiffuser/det/BEVDiffuser_tiny_onlyseg_sam3_metric3d/checkpoint-50000"
RUN_NAME="DiffBEVFormer_tiny_onlyseg_metric3d_studentbev_t100_alpha100_lambda0.55"
WORK_DIR="../results/diff_bevformer"

# Keep the existing Stage 2 deterministic setting. Forward every extra argument.
# Example: --resume_from /path/to/epoch_12.pth
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_onlyseg.py $CONFIG \
    --launcher pytorch \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --no-model-target \
    "$@"
