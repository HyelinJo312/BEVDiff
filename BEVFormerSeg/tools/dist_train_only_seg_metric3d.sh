#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# export ACCELERATE_MIXED_PRECISION=no
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


GPUS=4
PORT=${PORT:-28505}
CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_metric3d_v2.py"
# UNET_CHECKPOINT_DIR="../results/map_seg/bevdiffuser/BEVDiffuserSeg_tiny_onlyseg_sam3_metric3d/checkpoint-50000"
UNET_CHECKPOINT_DIR="../results/map_seg/bevdiffuser/BEVDiffuserSeg_tiny_onlyseg_sam3_metric3d_v2_2/checkpoint-50000"
LOAD_FROM="../results/map_seg/baseline/bevformer_tiny_seg_v3/epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_onlyseg_metric3d_v2_2_t100_alpha10_lambda0.55"
WORK_DIR="../results/map_seg/diff_bevformer"


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
    --load_from=$LOAD_FROM 
