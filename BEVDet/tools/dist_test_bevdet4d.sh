#!/usr/bin/env bash
# Evaluate a BEVDet4D checkpoint (e.g. epoch_20_ema.pth) on nuScenes val (mAP/NDS).
#
# Always passes --no-aavt. Without it, tools/test.py switches every 4D model to
# align_after_view_transfromation=True (tools/test.py:211-213). That path's gen_grid
# (mmdet3d/models/detectors/bevdet.py:322) assumes a 3x3 bda and crashes on the 4x4 bda
# produced by the dev3.0 BEVAug. --no-aavt evaluates the same feature path used in training;
# the config header reports mAP 0.3157 / NDS 0.4485 for that setting.
#
# Run from BEVDiff/BEVDet with jhl-bevdet:
#   bash tools/dist_test_bevdet4d.sh ../results/bevdet/det/bevdet4d-r50-cbgs_seed0/epoch_20_ema.pth
#   DRY_RUN=1 bash tools/dist_test_bevdet4d.sh <ckpt>          # print the command only
# Env overrides: CUDA_VISIBLE_DEVICES, GPUS (default: number of visible GPUs), CONFIG, OUT_PREFIX, PORT.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT=${1:?usage: dist_test_bevdet4d.sh <checkpoint> [extra test.py args]}
shift
CONFIG=${CONFIG:-./configs/bevdet/bevdet-r50-4d-cbgs.py}
GPUS=${GPUS:-$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}
PORT=${PORT:-28621}
OUT_PREFIX=${OUT_PREFIX:-"$(dirname "$CHECKPOINT")/eval_$(basename "$CHECKPOINT" .pth)"}
DRY_RUN=${DRY_RUN:-0}

[ -f "$CHECKPOINT" ] || { echo "error: missing $CHECKPOINT" >&2; exit 1; }
if [ -e "$OUT_PREFIX" ]; then
    echo "error: $OUT_PREFIX exists; set OUT_PREFIX to a new path" >&2
    exit 1
fi

python -c "import mmdet3d, sys; sys.exit(0 if mmdet3d.__version__.startswith('1.0.0') else 1)" || {
    echo "error: activate jhl-bevdet (BEVDet mmdet3d 1.0.0rc4 fork) first" >&2; exit 1; }

CMD=(torchrun --nproc_per_node="$GPUS" --master_port="$PORT"
     tools/test.py "$CONFIG" "$CHECKPOINT"
     --launcher pytorch
     --no-aavt
     --eval bbox
     --eval-options "jsonfile_prefix=$OUT_PREFIX"
     "$@")

if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "$OUT_PREFIX"
"${CMD[@]}" 2>&1 | tee "$OUT_PREFIX/test_log.txt"
