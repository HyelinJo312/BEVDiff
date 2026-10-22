#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG="./projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_v2.py"
CHECKPOINT="./results/diffbevformer/DiffBEVFormer_tiny_onlyseg_sam3_v2/epoch_6.pth"
GPUS=4
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
