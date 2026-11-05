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

# Rerun of the BEVDiffuser baseline with the detection-trained teacher -> map segmentation student.
#   teacher : BEVDiffuser_BEVFormer_tiny_original_bs4 (GT layout condition, Stage 1 on det BEVFormer-tiny)
#   target  : detection BEVFormer-tiny (cfg.model_target, loaded strictly from LOAD_FROM),
#             images re-normalized caffe BGR -> torchvision RGB, kept at the student's 0.8 scale
#             (model_target_img_rescale=1.0, matching the Ours det->seg run; the config default is 0.625)
#   stage 2 : t=5, 5 DDIM steps, CFG 2 (fixed in tools/bevdiffuser.py), MSE x10 distillation
CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_original.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_tiny_original_bs4/checkpoint-50000"
LOAD_FROM="../BEVFormer/ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_seg_original_renorm"

WORK_DIR="../results/diff_bevformer0/map_seg/baseline"

# RESUME_FROM="${WORK_DIR}/${RUN_NAME}/latest.pth"

PYTHON="${PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import mmcv, mmdet3d" >/dev/null 2>&1; then
    echo "error: $PYTHON cannot import mmcv/mmdet3d; activate the conda env first" >&2
    exit 1
fi

torchrun --nproc_per_node $GPUS \
    --master_port=$PORT \
    "$(dirname "$0")/train_layout.py" $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir $WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='BEVFormerSeg' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    --cfg-options model_target_img_rescale=1.0
    # --resume_from=$RESUME_FROM
