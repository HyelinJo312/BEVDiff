#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_NUM_THREADS=1
export MAX_JOBS=16
export TCNN_CUDA_ARCHITECTURES=86

GPUS=${1:-1}
PORT=${PORT:-28509}

CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_v4_adapter_aux.py"
CHECKPOINT="../results/version2/stage2/DiffBEVFormer_tiny_seg_v8/latest.pth"
OUT_DIR="../results/bev_visualize_stage2"
MAX_SAMPLES=-1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
    $(dirname "$0")/visualize_bev.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --out-dir ${OUT_DIR} \
    --max-samples ${MAX_SAMPLES} \
    --agg l1 --smooth-sigma 0.8 \
    --joint-clip-low 2.0 --joint-clip-high 98.0 \
    --gamma 1.0 --cmap viridis --interp bilinear \
    --dpi 300
