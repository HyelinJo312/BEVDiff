#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPUS="${GPUS:-4}"
PORT="${PORT:-28509}"
CONFIG="$ROOT/projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_metric3d.py"
RUN_NAME="DiffBEVFormerSeg_metric3d_t300_cfg4_lambda055"
WORK_DIR="$ROOT/results/diffbevformer/$RUN_NAME"
UNET_CHECKPOINT_DIR="${UNET_CHECKPOINT_DIR:-$ROOT/results/bevdiffuser/stage1/BEVDiffuserSeg_tiny_onlyseg_sam3_metric3d/checkpoint-50000}"
# --load_from initializes the frozen segmentation BEV target in train_onlyseg.py.
LOAD_FROM="${LOAD_FROM:-$ROOT/results/bevformer_tiny_seg_v3/epoch_24.pth}"

if [[ ! -f "$LOAD_FROM" ]]; then
    echo "Missing segmentation BEV checkpoint: $LOAD_FROM" >&2
    echo "Set LOAD_FROM to the trained BEVFormerSeg checkpoint." >&2
    exit 1
fi
if [[ ! -f "$UNET_CHECKPOINT_DIR/unet/diffusion_pytorch_model.safetensors" \
      && ! -f "$UNET_CHECKPOINT_DIR/diffusion_pytorch_model.safetensors" \
      && ! -f "$UNET_CHECKPOINT_DIR" ]]; then
    echo "Missing Stage 1 UNet checkpoint: $UNET_CHECKPOINT_DIR" >&2
    echo "Set UNET_CHECKPOINT_DIR to the Stage 1 checkpoint directory." >&2
    exit 1
fi

torchrun --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$ROOT/tools/train_onlyseg.py" "$CONFIG" \
    --launcher pytorch --deterministic \
    --work_dir "$WORK_DIR" \
    --unet_checkpoint_dir "$UNET_CHECKPOINT_DIR" \
    --load_from "$LOAD_FROM" \
    --report_to tensorboard \
    --tracker_project_name BEVFormerSeg \
    --tracker_run_name "$RUN_NAME" \
    "$@"
