#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

# export NVIDIA_TF32_OVERRIDE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_NUM_THREADS=1       # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=4
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/bev_tiny_dino_seg_sam.py"
UNET_CHECKPOINT_DIR="../results/version2/stage1/BEVDiffuser_tiny_dinoseg_v3_sam/checkpoint-50000"
# LOAD_FROM="./ckpts/bevformer_r101_dcn_24ep.pth"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_dinoseg_mgd_v5"
WORK_DIR="../results/version2/stage2"

# RESUME_FROM="../results/version2/stage2/DiffBEVFormer_tiny_dinoseg_mgd_v4/epoch_6.pth"
# export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
# taskset: GPU 4-7 are on NUMA node 1 -> pin to its local CPUs (16-31,48-63) for memory locality
taskset -c 16-31,48-63 torchrun --nproc_per_node=$GPUS --master_port=29507 \
    $(dirname "$0")/train_dino.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='wandb' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \
