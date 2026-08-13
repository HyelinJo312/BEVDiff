#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_FOR_THREADS_NUM=1   # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
                                  # OPENCV_NUM_THREADS는 OpenCV가 읽지 않는 변수라 무시된다
                                  # (확인: 설정해도 cv2.getNumThreads()==64)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=4
PORT=${PORT:-28506}

CONFIG="./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_v2.py"
UNET_CHECKPOINT_DIR="../results/version2/stage1/BEVDiffuser_tiny_onlyseg_sam3_v2/checkpoint-50000"
LOAD_FROM="./ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_onlyseg_sam_mgd_v4_t100_alpha100_lambda_0.6"
WORK_DIR="../results/version2/stage2"

# RESUME_FROM="../results/version2/stage2/DiffBEVFormer_tiny_sam_mgd_t10_alpha100_lambda_0.65/epoch_12.pth"

# export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_er_node=$GPUS --master_port=$PORT \
# taskset -c 0-15,32-47 torchrun --nproc_per_node=$GPUS \
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_onlyseg.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='wandb' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    # --resume_from=$RESUME_FROM \