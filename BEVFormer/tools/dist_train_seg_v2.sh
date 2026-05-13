#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_NUM_THREADS=1       # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=$1
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_v4_proj_fgmask.py"
UNET_CHECKPOINT_DIR="../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RESUME_FROM="None"
RUN_NAME="DiffBEVFormer_tiny_seg_v17"
WORK_DIR="../results/version2/stage2"

# export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
torchrun --nproc_per_node=4 --master_port=29503 \
    $(dirname "$0")/train_seg.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='wandb' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \
