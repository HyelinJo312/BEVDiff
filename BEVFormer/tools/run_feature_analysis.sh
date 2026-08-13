#!/usr/bin/env bash
# Run BEV feature distribution analysis (Hypothesis D).
# Compares baseline (BEVDiffuser, layout-only Stage-1) vs ours (Semantic
# Guidance Stage-1) on paired val frames.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

NUM_FRAMES=${NUM_FRAMES:-200}
OUTPUT_DIR=${OUTPUT_DIR:-../results/feature_analysis/v3_2_vs_baseline}

CONFIG_BASELINE="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/layout_tiny.py"
CKPT_BASELINE="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/epoch_24.pth"
CONFIG_OURS="../results/version2/stage2/DiffBEVFormer_tiny_seg_v5/layout_tiny_seg_v4_adapter_v2.py"
CKPT_OURS="../results/version2/stage2/DiffBEVFormer_tiny_seg_v5/epoch_24.pth"
TEACHER_INIT="./ckpts/bevformer_tiny_epoch_24.pth"

python tools/analyze_bev_distribution.py \
    --config_baseline "$CONFIG_BASELINE" \
    --ckpt_baseline   "$CKPT_BASELINE" \
    --config_ours     "$CONFIG_OURS" \
    --ckpt_ours       "$CKPT_OURS" \
    --teacher_init_ckpt "$TEACHER_INIT" \
    --num_frames "$NUM_FRAMES" \
    --output_dir "$OUTPUT_DIR"
