#!/usr/bin/env bash
# BEVDet4D-R50 baseline WITHOUT CBGS (C0, no-CBGS recipe; added 2026-11-05), trained from scratch:
#   configs/bevdet/bevdet-r50-4d.py = bevdet-r50-4d-cbgs.py without the CBGSDataset wrapper, 24 epochs,
#   lr step [24] (upstream non-CBGS bevdet-r50.py schedule); 4 GPUs x 8 = batch 32 (as the CBGS C0 run),
#   AdamW lr 2e-4 / wd 1e-2, MEGVIIEMAHook (EMA), SequentialControlHook (previous frame from the 4th epoch on).
#   28,130 train samples -> 879 iterations/epoch (CBGS: 3,862).
# Its EMA weights (epoch_24_ema.pth) are the frozen clean target for RECIPE=nocbgs in
#   projects/bevdiffuser/train_only_seg_cache_det_bevdet4d.sh (Stage 1) and tools/dist_train_diff_bevdet4d.sh (Stage 2).
#
# Run from BEVDiff/BEVDet with the jhl-bevdet env (mmdet3d 1.0.0rc4 fork; not jhl-bevdiff):
#   bash tools/dist_train_bevdet4d_r50_nocbgs.sh                    # new run
#   RESUME_EPOCH=7 bash tools/dist_train_bevdet4d_r50_nocbgs.sh     # resume model/optimizer AND EMA from epoch 7
#   DRY_RUN=1 bash tools/dist_train_bevdet4d_r50_nocbgs.sh          # print the command only
# max_keep_ckpts=1 in the config keeps only the latest epoch_N.pth, so only the latest epoch can be resumed;
#   every epoch_N_ema.pth is kept.
# Evaluate: CONFIG=configs/bevdet/bevdet-r50-4d.py bash tools/dist_test_bevdet4d.sh <WORK_DIR>/epoch_24_ema.pth
# Env overrides: CUDA_VISIBLE_DEVICES (default 4,5,6,7), SEED (default 0), RUN_NAME, WORK_DIR, PORT.
# Note: with RESUME_EPOCH, do not pass another --cfg-options in "$@" (it would replace the EMA resume option).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4   # batch 32 (= 4 x samples_per_gpu 8), same as the CBGS C0 run
PORT=${PORT:-28621}
SEED=${SEED:-0}
CONFIG="./configs/bevdet/bevdet-r50-4d.py"
RUN_NAME=${RUN_NAME:-"bevdet4d-r50-nocbgs_seed${SEED}"}
WORK_DIR=${WORK_DIR:-"../results/bevdet/det/${RUN_NAME}"}
RESUME_EPOCH=${RESUME_EPOCH:-}
DRY_RUN=${DRY_RUN:-0}

NVIS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if [ "$NVIS" -ne "$GPUS" ]; then
    echo "error: this recipe needs $GPUS GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has $NVIS" >&2
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
