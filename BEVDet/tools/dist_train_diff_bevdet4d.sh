#!/usr/bin/env bash
# Stage 2: DiffBEVDet4D-R50 (student from scratch, C0 recipe) distilled from the semantic-only Stage 1
# teacher. RECIPE (required) picks the C0 recipe and everything that must match it:
#   RECIPE=nocbgs  projects/configs/diff_bevdet/diff_bevdet4d_r50_nocbgs_onlyseg_mgd.py  (no CBGS, 24 epochs)
#                  target bevdet4d-r50-nocbgs_seed0/epoch_24_ema.pth,
#                  teacher BEVDiffuser_bevdet4d_r50_nocbgs_onlyseg_sam3_metric3d/checkpoint-50000
#   RECIPE=cbgs    projects/configs/diff_bevdet/diff_bevdet4d_r50_cbgs_onlyseg_mgd.py    (CBGS, 20 epochs)
#                  target bevdet4d-r50-cbgs_seed0/epoch_20_ema.pth,
#                  teacher BEVDiffuser_bevdet4d_r50_cbgs_onlyseg_sam3_metric3d/checkpoint-50000
#   Both: 4 GPUs x 8 = batch 32 (as the C0 runs), teacher t=100 / DDIM 5 / CFG 2,
#   MGD lambda 0.55 alpha 100, loss 0.5*task + 0.5*bev.
#   TARGET_CKPT (load_from) initialises ONLY the frozen BEV target; the student starts from
#   init_weights (ImageNet ResNet-50), as in the flagship DiffBEVFormer run.
#
# Run from BEVDiff/BEVDet with the jhl-bevdet env:
#   RECIPE=nocbgs bash tools/dist_train_diff_bevdet4d.sh                  # new run
#   RECIPE=nocbgs RESUME_EPOCH=7 bash tools/dist_train_diff_bevdet4d.sh   # resume model/optimizer AND EMA
#   RECIPE=nocbgs DRY_RUN=1 bash tools/dist_train_diff_bevdet4d.sh        # print the command only
# Evaluate: CONFIG=<the RECIPE's config> bash tools/dist_test_bevdet4d.sh <WORK_DIR>/epoch_<last>_ema.pth
# Env overrides: CUDA_VISIBLE_DEVICES (default 4,5,6,7), SEED (default 0), RUN_NAME, WORK_DIR, PORT,
#   TARGET_CKPT, UNET_CKPT.
# Note: with RESUME_EPOCH, do not pass another --cfg-options in "$@" (it would replace the EMA resume option).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

GPUS=4   # batch 32 (= 4 x samples_per_gpu 8), same as the C0 runs
PORT=${PORT:-28630}
SEED=${SEED:-0}
RECIPE=${RECIPE:-}
case "$RECIPE" in
    nocbgs) C0_RUN="bevdet4d-r50-nocbgs_seed0"; LAST_EPOCH=24 ;;
    cbgs)   C0_RUN="bevdet4d-r50-cbgs_seed0";   LAST_EPOCH=20 ;;
    *) echo "error: set RECIPE=nocbgs or RECIPE=cbgs (C0 recipe), got '$RECIPE'" >&2; exit 1 ;;
esac
CONFIG="./projects/configs/diff_bevdet/diff_bevdet4d_r50_${RECIPE}_onlyseg_mgd.py"
TARGET_CKPT=${TARGET_CKPT:-"../results/bevdet/det/${C0_RUN}/epoch_${LAST_EPOCH}_ema.pth"}
UNET_CKPT=${UNET_CKPT:-"../results/bevdiffuser/det/BEVDiffuser_bevdet4d_r50_${RECIPE}_onlyseg_sam3_metric3d/checkpoint-50000"}
RUN_NAME=${RUN_NAME:-"diff_bevdet4d-r50-${RECIPE}_onlyseg_mgd_seed${SEED}"}
WORK_DIR=${WORK_DIR:-"../results/diff_bevdet/det/${RUN_NAME}"}
RESUME_EPOCH=${RESUME_EPOCH:-}
DRY_RUN=${DRY_RUN:-0}

NVIS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if [ "$NVIS" -ne "$GPUS" ]; then
    echo "error: this recipe needs $GPUS GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has $NVIS" >&2
    exit 1
fi
[ -f "$TARGET_CKPT" ] || { echo "error: missing TARGET_CKPT $TARGET_CKPT (train C0 first)" >&2; exit 1; }
[ -f "$UNET_CKPT/unet/diffusion_pytorch_model.safetensors" ] || [ -f "$UNET_CKPT/diffusion_pytorch_model.safetensors" ] || [ -f "$UNET_CKPT" ] || {
    echo "error: missing Stage 1 UNet in UNET_CKPT $UNET_CKPT (train Stage 1 first)" >&2; exit 1; }

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
     tools/train_onlyseg.py "$CONFIG"
     --launcher pytorch
     --seed "$SEED"
     --work-dir "$WORK_DIR"
     --load_from "$TARGET_CKPT"
     --unet_checkpoint_dir "$UNET_CKPT"
     "${EXTRA[@]}" "$@")

if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "$WORK_DIR"
{
    echo "date: $(date '+%F %T %Z')  host: $(hostname)"
    echo "cmd: $(printf '%q ' "${CMD[@]}")"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  RECIPE=$RECIPE  python=$(command -v python)"
    echo "BEVDet HEAD: $(git rev-parse HEAD)"
    echo "BEVDet working tree (git status --short):"
    git status --short
} > "$WORK_DIR/launch_$(date +%Y%m%d_%H%M%S).txt"

"${CMD[@]}"
