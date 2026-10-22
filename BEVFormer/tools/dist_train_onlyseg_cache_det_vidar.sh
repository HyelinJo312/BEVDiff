#!/usr/bin/env bash
set -euo pipefail

# Run from BEVFormer/. ViDAR target and student initialization are specified
# independently in the config; --load_from is reserved for the legacy teacher.
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4
PORT=${PORT:-28508}
CONFIG='./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_metric3d_vidar.py'
UNET_CHECKPOINT_DIR='../results/bevdiffuser/BEVDiffuser_tiny_onlyseg_sam3_metric3d_vidar/checkpoint-50000'
RUN_NAME='DiffBEVFormer_tiny_onlyseg_metric3d_vidar_t100_alpha100_lambda_0.55_50k'
# train_onlyseg.py appends RUN_NAME to --work_dir, yielding a unique run dir.
WORK_DIR='../results/diff_bevformer'

torchrun --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$(dirname "$0")/train_onlyseg.py" "$CONFIG" \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir="$WORK_DIR" \
    --report_to=tensorboard \
    --tracker_project_name=DiffBEVFormer \
    --tracker_run_name="$RUN_NAME" \
    --unet_checkpoint_dir="$UNET_CHECKPOINT_DIR"
