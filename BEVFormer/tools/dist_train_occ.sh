#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# Avoid CPU oversubscription across 4 trainers and 32 data-loader workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

GPUS=8
PORT=28506
CONFIG="./projects/configs/occ3d/bevformer_base_occ.py"
RUN_NAME="bevformer_base_occ_v4"
WORK_DIR="../results/occ3d"
RESUME_FROM="${RESUME_FROM:-${WORK_DIR}/${RUN_NAME}/latest.pth}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..":$PYTHONPATH
cd "${SCRIPT_DIR}/.."

# if [[ ! -f "${RESUME_FROM}" ]]; then
#     echo "Resume checkpoint not found: ${RESUME_FROM}" >&2
#     exit 1
# fi

torchrun --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/train_occ.py" "${CONFIG}" \
    --launcher pytorch \
    --deterministic \
    --work_dir="${WORK_DIR}" \
    --report_to='tensorboard' \
    --tracker_project_name='Occ3D-BEVFormer' \
    --tracker_run_name="${RUN_NAME}" \
    --resume_from="${RESUME_FROM}"
