#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_FOR_THREADS_NUM=1   # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=4
PORT=${PORT:-28507}

CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_v4_sam_mgd.py"
UNET_CHECKPOINT_DIR="../results/version2/stage1/BEVDiffuser_tiny_sam3_v1/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_sam_mgd_t100_alpha100_lambda_0.6_v2"
WORK_DIR="../results/version2/stage2"

# RESUME_FROM="../results/version2/stage2/DiffBEVFormer_tiny_sam_mgd_t100_alpha100_lambda_0.6/epoch_12.pth"

# export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_er_node=$GPUS --master_port=$PORT \
# taskset -c 16-31,48-63 torchrun --nproc_per_node $GPUS \
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_seg.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \