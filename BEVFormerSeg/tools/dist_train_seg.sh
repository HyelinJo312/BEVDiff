#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1          # DDP: 프로세스간 thread 경합 방지 (PyTorch DDP 권장)
export MKL_NUM_THREADS=1          # DDP: 동일 이유
export NUMEXPR_NUM_THREADS=1      # DDP: 동일 이유
export OPENBLAS_NUM_THREADS=1     # DDP: 동일 이유
export OPENCV_NUM_THREADS=1       # DataLoader worker 16개 × 2 = 32 threads ≒ 가용 CPU 32개
export MAX_JOBS=16                # parallel build jobs (32 CPUs available)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000 = sm_86)

PORT=${PORT:-28509}

CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_det_semantic_v4_2.py"
UNET_CHECKPOINT_DIR="./results/bevdiffuser/stage1/BEVDiffuserSeg_tiny_v3_task-loss_01_attn421_segmap2/checkpoint-50000"
LOAD_FROM="./results/bevformer_tiny_seg_v3/epoch_24.pth"
# RESUME_FROM="./results/version2/stage2/DiffBEVFormer_tiny_seg_FDN_proj/latest.pth"
RUN_NAME="DiffBEVFormer_tiny_seg_FDN_v2"
WORK_DIR="./results/diffbevformer"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=4 --master_port=29505 \
    $(dirname "$0")/train_seg.py $CONFIG \
    --launcher pytorch \
    --deterministic \
    --work_dir $WORK_DIR \
    --unet_checkpoint_dir $UNET_CHECKPOINT_DIR \
    --load_from $LOAD_FROM \
    --report_to='tensorboard' \
    --tracker_project_name='BEVFormerSeg' \
    --tracker_run_name=$RUN_NAME
    # --resume_from=$RESUME_FROM

