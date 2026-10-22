#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no
# Silence import-time warnings re-printed by every spawned DataLoader worker.
export PYTHONWARNINGS="ignore:On January 1:UserWarning,ignore:The 'nopython' keyword"


GPUS=4
PORT=${PORT:-28509}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="./projects/configs/bevformer/bevformer_tiny_seg_v3.py"
RUN_NAME="BEVFormerSeg_tiny_v3"
WORK_DIR="../results/map_seg/baseline/bevformer_tiny_seg_v3"
RESUME_FROM="../results/map_seg/baseline/bevformer_tiny_seg_v3/epoch_12.pth"

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT 
# torchrun --nproc_per_node=4

PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
torchrun --nproc_per_node $GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir $WORK_DIR \
    --report_to='tensorboard' \
    --tracker_project_name='BEVFormerSeg' \
    --tracker_run_name=$RUN_NAME \
    --resume-from=$RESUME_FROM \
