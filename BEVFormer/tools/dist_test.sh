#!/usr/bin/env bash

CONFIG="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/layout_tiny.py"
CHECKPOINT="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/epoch_24.pth"
GPUS=$3
PORT=${PORT:-29503}

# export PYTHONWARNINGS="ignore"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=4 --master_port=29505 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    # --format_only \
