#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export OPENCV_NUM_THREADS=1
export MAX_JOBS=16
export TCNN_CUDA_ARCHITECTURES=86

GPUS=4
PORT=28506
CONFIG="./projects/configs/occ3d/bevformer_base_occ3d.py"
RUN_NAME="BEVFormer_base_occ3d"
WORK_DIR="../results/occ3d"
CPU_CORES="16-31,48-63"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..":$PYTHONPATH

taskset -c "${CPU_CORES}" torchrun \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/train_occ.py" "${CONFIG}" \
    --launcher pytorch \
    --deterministic \
    --work_dir="${WORK_DIR}" \
    --report_to='tensorboard' \
    --tracker_project_name='Occ3D-BEVFormer' \
    --tracker_run_name="${RUN_NAME}" 
