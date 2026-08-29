#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_FOR_THREADS_NUM=1   # 1=single thread to avoid CPU thread explosione
export MAX_JOBS=16                # parallel build jobs
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

GPUS=${GPUS:-4}
PORT=${PORT:-28509}

CONFIG="./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_v2_da3_onestep.py"
UNET_CHECKPOINT_DIR="../results/version2/stage1/BEVDiffuser_tiny_onlyseg_sam3_v5/checkpoint-50000"
RUN_NAME="OneStepDiffBEVFormer_tiny_onlyseg_sam_da3_t100_step5_scratch"
WORK_DIR="../results/version2/stage2"

# RESUME_FROM="../results/version2/stage2/${RUN_NAME}/latest.pth"

export PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-}

taskset -c 16-31,48-63 torchrun --nproc_per_node "$GPUS" \
    --master_port="$PORT" \
    "$(dirname "$0")/train_onlyseg_onestep.py" "$CONFIG" \
    --launcher pytorch "${@:1}" \
    --deterministic \
    --work_dir="$WORK_DIR" \
    --report_to='tensorboard' \
    --tracker_project_name='SemBEV-Diff' \
    --tracker_run_name="$RUN_NAME" \
    --unet_checkpoint_dir="$UNET_CHECKPOINT_DIR"
    # --resume_from="$RESUME_FROM"
