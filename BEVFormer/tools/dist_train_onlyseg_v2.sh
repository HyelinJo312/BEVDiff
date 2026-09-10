#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_FOR_THREADS_NUM=1   # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=4
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_v2_da3_ablation1.py"
UNET_CHECKPOINT_DIR="../results/version2/stage1/BEVDiffuser_tiny_onlyseg_ablation1_concat/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_onlyseg_ablation1_concat_t100_alpha100_lambda_0.55"
WORK_DIR="../results/version2/stage2"

# RESUME_FROM="../results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_da3_no-mgd/epoch_18.pth"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# torchrun --nproc_per_node=4 --master_port=29507 \
# taskset -c 0-15,32-47 torchrun --nproc_per_node $GPUS \
taskset -c 16-31,48-63 torchrun --nproc_per_node $GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_onlyseg.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='SemBEV-Diff' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \