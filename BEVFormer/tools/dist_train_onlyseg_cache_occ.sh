#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# export ACCELERATE_MIXED_PRECISION=no
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

GPUS=8
PORT=${PORT:-28505}
CONFIG="./projects/configs/diff_bevformer/bev_onlyseg_sam_metric3d_occ_v2.py"
# Matches train_only_seg_cache.sh; override to select an existing earlier checkpoint.
UNET_CHECKPOINT_DIR="../results/bevdiffuser/occ/BEVDiffuser_occ_onlyseg_sam3_metric3d_v4/checkpoint-50000"
LOAD_FROM="../results/diff_bevformer/occ/baseline/bevformer_base_occ/epoch_24.pth"

RUN_NAME="DiffBEVFormer_occ_onlyseg_metric3d_v4_alpha0.1_lambda_0.55_t300_cfg4"
WORK_DIR="../results/diff_bevformer/occ"

GRADIENT_MONITOR_ENABLED=True
GRADIENT_MONITOR_INTERVAL=200

# RESUME_FROM="../results/diff_bevformer/occ/DiffBEVFormer_occ_onlyseg_metric3d_t100_alpha1_lambda_0.7/latest.pth"

# Keep the existing Stage 2 deterministic setting. Forward every extra argument.
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_onlyseg_occ.py $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    --gradient-monitor-enabled="$GRADIENT_MONITOR_ENABLED" \
    --gradient-monitor-interval="$GRADIENT_MONITOR_INTERVAL"
    # --resume_from=$RESUME_FROM
