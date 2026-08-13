#!/usr/bin/env bash

# Per-scene-condition nuScenes evaluation (overall / sunny / rainy / day / night).
# Runs a single inference pass, then reports mAP / NDS for each condition.

CONFIG="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/layout_tiny.py"
CHECKPOINT="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch/epoch_24.pth"
GPUS=${GPUS:-4}
PORT=${PORT:-29506}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_by_condition.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:1}
