#!/usr/bin/env bash
# BEVDet-R50 baseline (configs/bevdet/bevdet-r50.py: 24 ep, no CBGS, 8 GPUs x 8 = 64, lr 2e-4).
# Run from BEVDiff/BEVDet with the jhl-bevdet env (mmdet3d 1.0.0rc4 fork; not jhl-bevdiff):
#   conda activate jhl-bevdet && bash tools/dist_train_bevdet_r50.sh
# Also the frozen target / Stage 1 source for the BEVDet experiments.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=8
PORT=${PORT:-28610}
CONFIG="./configs/bevdet/bevdet-r50.py"
RUN_NAME="bevdet-r50_baseline"
WORK_DIR="../results/bevdet/det/${RUN_NAME}"

python -c "import mmdet3d, sys; sys.exit(0 if mmdet3d.__version__.startswith('1.0.0') else 1)" || {
    echo "error: activate jhl-bevdet (BEVDet mmdet3d 1.0.0rc4 fork) first" >&2; exit 1; }

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py $CONFIG \
    --launcher pytorch \
    --seed 0 \
    --work-dir $WORK_DIR "$@"
