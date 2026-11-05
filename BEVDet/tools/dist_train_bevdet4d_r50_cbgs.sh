#!/usr/bin/env bash
# BEVDet4D-R50-CBGS baseline, trained from scratch with the official recipe, config unchanged:
#   configs/bevdet/bevdet-r50-4d-cbgs.py: CBGS, 20 epochs, 8 GPUs x 8 = batch 64, AdamW lr 2e-4 / wd 1e-2,
#   MEGVIIEMAHook (EMA), SequentialControlHook (previous frame used from the 4th epoch on).
#   Reference in the config header (eval with align_after_view_transfromation=False): mAP 0.3157 / NDS 0.4485.
# Its EMA weights (epoch_20_ema.pth) are the frozen clean target / Stage 1 source for the BEVDet4D experiments.
#
# Run from BEVDiff/BEVDet with the jhl-bevdet env (mmdet3d 1.0.0rc4 fork; not jhl-bevdiff):
#   bash tools/dist_train_bevdet4d_r50_cbgs.sh                    # new run
#   RESUME_EPOCH=7 bash tools/dist_train_bevdet4d_r50_cbgs.sh     # resume model/optimizer AND EMA from epoch 7
#   DRY_RUN=1 bash tools/dist_train_bevdet4d_r50_cbgs.sh          # print the command only
# Env overrides: CUDA_VISIBLE_DEVICES, SEED (default 0), RUN_NAME, WORK_DIR, PORT.
# Note: with RESUME_EPOCH, do not pass another --cfg-options in "$@" (it would replace the EMA resume option).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4   # batch 64 (= 8 x samples_per_gpu 8) is part of the recipe: lr, EMA init_updates, iterations/epoch
PORT=${PORT:-28620}
SEED=${SEED:-0}
CONFIG="./configs/bevdet/bevdet-r50-4d-cbgs.py"
RUN_NAME=${RUN_NAME:-"bevdet4d-r50-cbgs_seed${SEED}"}
WORK_DIR=${WORK_DIR:-"../results/bevdet/det/${RUN_NAME}"}
RESUME_EPOCH=${RESUME_EPOCH:-}
DRY_RUN=${DRY_RUN:-0}

NVIS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if [ "$NVIS" -ne "$GPUS" ]; then
    echo "error: the official recipe needs $GPUS GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has $NVIS" >&2
    exit 1
fi

python -c "import mmdet3d, sys; sys.exit(0 if mmdet3d.__version__.startswith('1.0.0') else 1)" || {
    echo "error: activate jhl-bevdet (BEVDet mmdet3d 1.0.0rc4 fork) first" >&2; exit 1; }

EXTRA=()
if [ -n "$RESUME_EPOCH" ]; then
    CKPT="$WORK_DIR/epoch_${RESUME_EPOCH}.pth"
    EMA="$WORK_DIR/epoch_${RESUME_EPOCH}_ema.pth"
    for f in "$CKPT" "$EMA"; do
        [ -f "$f" ] || { echo "error: missing $f" >&2; exit 1; }
    done
    # resume_from alone restores model/optimizer but restarts the EMA; the hook needs its own resume path.
    EXTRA+=(--resume-from "$CKPT" --cfg-options "custom_hooks.0.resume=$EMA")
elif compgen -G "$WORK_DIR/epoch_*.pth" > /dev/null; then
    echo "error: $WORK_DIR already has checkpoints; set RESUME_EPOCH=<n> or a new RUN_NAME" >&2
    exit 1
fi

CMD=(torchrun --nproc_per_node="$GPUS" --master_port="$PORT"
     tools/train.py "$CONFIG"
     --launcher pytorch
     --seed "$SEED"
     --work-dir "$WORK_DIR"
     "${EXTRA[@]}" "$@")

if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "$WORK_DIR"
{
    echo "date: $(date '+%F %T %Z')  host: $(hostname)"
    echo "cmd: $(printf '%q ' "${CMD[@]}")"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  python=$(command -v python)"
    echo "BEVDet HEAD: $(git rev-parse HEAD)"
    echo "BEVDet working tree (git status --short):"
    git status --short
} > "$WORK_DIR/launch_$(date +%Y%m%d_%H%M%S).txt"

"${CMD[@]}"
