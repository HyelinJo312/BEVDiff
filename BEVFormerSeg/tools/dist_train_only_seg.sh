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

CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_v2_da3_1_4_nusc.py"
UNET_CHECKPOINT_DIR="./results/bevdiffuser/stage1/BEVDiffuserSeg_tiny_onlyseg_sam3_v4_1_4_nusc/checkpoint-30000"
LOAD_FROM="./results/bevformer_tiny_seg_v3_1_4_nusc/epoch_24.pth"

# RESUME_FROM="results/diffbevformer/DiffBEVFormer_tiny_onlyseg_sam3_v2/epoch_6.pth"

RUN_NAME="DiffBEVFormer_tiny_onlyseg_sam3_v8_1_4_nusc_t100_lambda_0.6"
WORK_DIR="./results/diffbevformer"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# torchrun --nproc_per_node=4 --master_port=29505 \
# taskset -c 16-31,48-63 torchrun --nproc_per_node $GPUS \
taskset -c 0-15,32-47 torchrun --nproc_per_node $GPUS \
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

