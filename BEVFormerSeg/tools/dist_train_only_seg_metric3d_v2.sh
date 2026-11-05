#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# export ACCELERATE_MIXED_PRECISION=no
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


# Detection-trained diffusion teacher -> map segmentation student (run from BEVFormerSeg/).
#   teacher : BEVDiffuser_tiny_onlyseg_sam3_metric3d (det taxonomy, Stage-1 config
#             BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py)
#   target  : detection BEVFormer-tiny (cfg.model_target, loaded strictly from LOAD_FROM),
#             images re-normalized caffe BGR -> torchvision RGB, kept at the student's 0.8 scale
#   stage 2 : noise/denoise t=300, 5 DDIM steps, CFG 4, MGD alpha 10 / lambda 0.55 (set in CONFIG)
GPUS=4
PORT=${PORT:-28505}
CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_metric3d_det.py"
UNET_CHECKPOINT_DIR="../results/bevdiffuser/det/BEVDiffuser_tiny_onlyseg_sam3_metric3d/checkpoint-50000"
LOAD_FROM="../BEVFormer/ckpts/bevformer_tiny_epoch_24.pth"
RUN_NAME="DiffBEVFormer_tiny_onlyseg_metric3d_det_renorm_t300_cfg4_alpha10_lambda0.55"
WORK_DIR="../results/diff_bevformer0/map_seg"

RESUME_FROM="../results/diff_bevformer0/map_seg/DiffBEVFormer_tiny_onlyseg_metric3d_det_renorm_t300_cfg4_alpha10_lambda0.55/latest.pth"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_onlyseg.py $CONFIG \
    --launcher pytorch "$@" \
    --deterministic \
    --work_dir=$WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='DiffBEVFormer' \
    --tracker_run_name=$RUN_NAME \
    --unet_checkpoint_dir=$UNET_CHECKPOINT_DIR \
    --load_from=$LOAD_FROM \
    --resume_from=$RESUME_FROM
