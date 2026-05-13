#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_NUM_THREADS=1       # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=8                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=$1
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/layout_tiny.py"
UNET_CHECKPOINT_DIR="../results/BEVDiffuser_BEVFormer_tiny_original_bs2/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
# RESUME_FROM="../results/version2/stage2/DiffBEVFormer_tiny_original/epoch_12.pth"
RUN_NAME="DiffBEVFormer_tiny_original_24epoch"
WORK_DIR="../results/version2/stage2"

# export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
torchrun --nproc_per_node=4 --master_port=29503 \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \
