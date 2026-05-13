#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Sanity check: inject Teacher BEV into Student det head, measure NDS/mAP.
# Run from BEVFormer/ directory.
#
# Usage:
#   cd BEVFormer
#   bash tools/dist_test_sanity.sh [mode] [gpus]
#
# Modes:
#   student                      — Student baseline (no Teacher, just reference)
#   teacher_bev                  — Teacher raw BEV → Student det head (no diffuser)
#   teacher_diffuser             — Teacher+Diffuser GT BEV → Student det head
#   teacher_diffuser_layout_only — Teacher+Diffuser with layout cond ONLY (no seg/depth)
#   teacher_self      — Teacher BEV → Teacher det head (upper bound)
#
# Example:
#   bash tools/dist_test_sanity.sh teacher_diffuser 4

set -e

# ── GPU / NCCL ───────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=1

# ── Paths ────────────────────────────────────────────────────────────────────
# Config for full-guidance modes (seg + depth + layout)
CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_v4_adapter.py"
# CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_v4.py"  # no adapter
# Config for layout-only ablation (no seg/depth encoder in UNet)
CONFIG_LAYOUT_ONLY="./projects/configs/diff_bevformer/layout_tiny.py"

# Stage-2 Student checkpoint (output of train_seg.py stage 2)
STUDENT_CKPT="../results/version2/stage2/DiffBEVFormer_tiny_seg_v2/epoch_24.pth"    # with adapter
# STUDENT_CKPT="../results/version2/stage2/DiffBEVFormer_tiny_seg_v3/epoch_12.pth"  # no adapter
STUDENT_CKPT="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/epoch_12.pth"  # layout-only

# Stage-1 Teacher checkpoint (BEVFormer-tiny trained to epoch 24)
TEACHER_CKPT="./ckpts/bevformer_tiny_epoch_24.pth"

# Stage-1 UNet checkpoint dir — full guidance (seg+depth+layout)
UNET_CKPT_DIR="../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000"
# Stage-1 UNet checkpoint dir — layout-only baseline
UNET_CKPT_DIR_LAYOUT_ONLY="../results/BEVDiffuser_BEVFormer_tiny_original_bs2/checkpoint-50000"

# ── Args ─────────────────────────────────────────────────────────────────────
MODE=${1:-teacher_diffuser}   # student | teacher_bev | teacher_diffuser | teacher_diffuser_layout_only | teacher_self
GPUS=${2:-4}
PORT=${PORT:-29509}

# [teacher_diffuser_layout_only] Select config & UNet checkpoint based on mode
if [ "${MODE}" = "teacher_diffuser_layout_only" ]; then
    ACTIVE_CONFIG="${CONFIG_LAYOUT_ONLY}"
    ACTIVE_UNET="${UNET_CKPT_DIR_LAYOUT_ONLY}"
else
    ACTIVE_CONFIG="${CONFIG}"
    ACTIVE_UNET="${UNET_CKPT_DIR}"
fi

echo "========================================"
echo "  mode        : ${MODE}"
echo "  config      : ${ACTIVE_CONFIG}"
echo "  student ckpt: ${STUDENT_CKPT}"
echo "  teacher ckpt: ${TEACHER_CKPT}"
echo "  unet dir    : ${ACTIVE_UNET}"
echo "  GPUs        : ${GPUS}"
echo "========================================"

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" \
    "$(dirname "$0")"/test_sanity.py \
    "${ACTIVE_CONFIG}" \
    "${STUDENT_CKPT}" \
    --teacher_ckpt  "${TEACHER_CKPT}" \
    --unet_ckpt_dir "${ACTIVE_UNET}" \
    --mode "${MODE}" \
    --eval bbox \
    --launcher pytorch
