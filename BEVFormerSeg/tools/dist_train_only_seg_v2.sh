#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_FOR_THREADS_NUM=1   # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

PORT=${PORT:-28509}
GPUS=4

CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_v3.py"
UNET_CHECKPOINT_DIR="./results/bevdiffuser/stage1/BEVDiffuserSeg_tiny_onlyseg_sam3_v3/checkpoint-50000"
LOAD_FROM="./results/bevformer_tiny_seg_v3/epoch_24.pth"

# RESUME_FROM="results/diffbevformer/DiffBEVFormer_tiny_onlyseg_sam3_v2/epoch_6.pth"

RUN_NAME="DiffBEVFormer_tiny_onlyseg_sam3_v2"
WORK_DIR="./results/diffbevformer"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# torchrun --nproc_per_node=4 --master_port=29505 \
taskset -c 0-15,32-47 torchrun --nproc_per_node $GPUS \
    --master_port=29505 \
    $(dirname "$0")/train_onlyseg.py $CONFIG \
    --launcher pytorch \
    --deterministic \
    --work_dir $WORK_DIR \
    --unet_checkpoint_dir $UNET_CHECKPOINT_DIR \
    --load_from $LOAD_FROM \
    --report_to='tensorboard' \
    --tracker_project_name='BEVFormerSeg' \
    --tracker_run_name=$RUN_NAME \
    # --resume_from=$RESUME_FROM

