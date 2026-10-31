#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4
PORT=${PORT:-28509}

# BEVDiffuser teacher retrained on the segmentation BEV (Stage 1: projects/bevdiffuser/train_original.sh).
# x0 comes from the segmentation BEVFormer-tiny, so LOAD_FROM is the seg baseline.
CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_original_segteacher.py"
UNET_CHECKPOINT_DIR="../results/map_seg/bevdiffuser/BEVDiffuserSeg_tiny_original/checkpoint-50000"
LOAD_FROM="../results/map_seg/baseline/bevformer_tiny_seg_v3/epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_seg_original_segteacher"

WORK_DIR="../results/map_seg/diff_bevformer/baseline"

# RESUME_FROM="./results/bevformer_tiny_seg_v2/epoch_12.pth"

PYTHON="${PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import mmcv, mmdet3d" >/dev/null 2>&1; then
    echo "error: $PYTHON cannot import mmcv/mmdet3d; activate the conda env first" >&2
    exit 1
fi

"$PYTHON" -m torch.distributed.run --nproc_per_node $GPUS \
    --master_port=$PORT \
    "$(dirname "$0")/train_layout.py" $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir $WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='BEVFormerSeg' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM
    # --resume_from=$RESUME_FROM
