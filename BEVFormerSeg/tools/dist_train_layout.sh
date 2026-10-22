#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no


GPUS=4
PORT=${PORT:-28509}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_original.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_tiny_original_bs4/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_mapseg_epoch_24.pth"

RUN_NAME="DiffBEVFormer_tiny_seg_original"
WORK_DIR="../results/map_seg/diffbevformer/baseline"

# RESUME_FROM="./results/bevformer_tiny_seg_v2/epoch_12.pth"

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT 
# torchrun --nproc_per_node=4

PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
torchrun --nproc_per_node $GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_layout.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir $WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='BEVFormerSeg' \
    --tracker_run_name=$RUN_NAME \
    # --resume-from=$RESUME_FROM \
