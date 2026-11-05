#!/usr/bin/env bash
# Evaluate a BEVDet checkpoint (e.g. the EMA weights epoch_24_ema.pth) on nuScenes val (NDS/mAP).
# Run from BEVDiff/BEVDet with jhl-bevdet active:
#   bash tools/dist_test_bevdet.sh ../results/bevdet/det/bevdet-r50_baseline/epoch_24_ema.pth [config]
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT=$1
CONFIG=${2:-./configs/bevdet/bevdet-r50.py}
GPUS=${GPUS:-8}
PORT=${PORT:-28611}

python -c "import mmdet3d, sys; sys.exit(0 if mmdet3d.__version__.startswith('1.0.0') else 1)" || {
    echo "error: activate jhl-bevdet (BEVDet mmdet3d 1.0.0rc4 fork) first" >&2; exit 1; }

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --eval bbox \
    --eval-options jsonfile_prefix="$(dirname "$CHECKPOINT")/eval_$(basename "$CHECKPOINT" .pth)"
