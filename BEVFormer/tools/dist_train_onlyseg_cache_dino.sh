#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4
PORT=${PORT:-28507}
CONFIG="./projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_v2_metric3d_dino"

UNET_CHECKPOINT_DIR="../results/bevdiffuser/BEVDiffuser_tiny_onlyseg_sam3_metric3d_dino/checkpoint-50000"
if [[ ! -f "$UNET_CHECKPOINT_DIR/unet/diffusion_pytorch_model.safetensors" ]]; then
    echo "Stage 1 UNet checkpoint not found: $UNET_CHECKPOINT_DIR" >&2
    exit 1
fi
RUN_NAME="DiffBEVFormer_tiny_onlyseg_metric3d_dino_obsmask_t100_alpha100_lambda_0.55_u0.1"
WORK_DIR="../results/diff_bevformer"

# Keep the existing Stage 2 deterministic setting. Forward every extra argument.
# Example: --resume_from /path/to/epoch_12.pth
torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$SCRIPT_DIR/train_onlyseg_dino.py" $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR
