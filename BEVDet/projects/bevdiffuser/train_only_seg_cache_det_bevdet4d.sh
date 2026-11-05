#!/usr/bin/env bash
# Stage 1: semantic-only BEV diffusion teacher on frozen BEVDet4D-R50 (C0) features (env jhl-bevdet).
# Hyper-parameters mirror the flagship BEVDiffuser_tiny_onlyseg_sam3_metric3d (its tensorboard hparams):
#   UNet from scratch, 4 GPUs x 4 = batch 16, AdamW lr 1e-4 constant (wd 1e-2, clip 1.0), 50k steps,
#   prediction_type=sample (x0), uncond_prob 0.2 (batch level), task_loss_scale 0, seed 0, fp32.
# BEVDet-side differences (decided 2026-11-04): BEVDet4D train augmentation incl. BEV aug, no CBGS,
#   UNet 128 -> 64/32/16 core, semantic cache warped to the BEVDet frame at 64x64 (see the config).
# RECIPE (required) = the C0 recipe of the frozen target; it sets the default target and run name,
#   so teachers trained on different targets never share an output directory:
#   RECIPE=nocbgs  target bevdet4d-r50-nocbgs_seed0/epoch_24_ema.pth (tools/dist_train_bevdet4d_r50_nocbgs.sh)
#   RECIPE=cbgs    target bevdet4d-r50-cbgs_seed0/epoch_20_ema.pth   (tools/dist_train_bevdet4d_r50_cbgs.sh)
#   Stage 1 itself is identical for both (its own data never uses CBGS).
#
# Usage (any cwd; the script runs from BEVDiff/BEVDet):
#   RECIPE=nocbgs bash projects/bevdiffuser/train_only_seg_cache_det_bevdet4d.sh
#   RECIPE=nocbgs RESUME_FROM=../results/bevdiffuser/det/<RUN_NAME>/checkpoint-20000 bash ...   # resume
#   RECIPE=nocbgs DRY_RUN=1 bash ...                                                             # print the command
# Env overrides: CUDA_VISIBLE_DEVICES, BEV_CHECKPOINT, RUN_NAME, OUTPUT_DIR, PORT, MAX_TRAINING_STEPS,
#   CHECKPOINT_STEP, GPUS (smoke tests only: the recipe is 4 GPUs x 4).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../.."   # BEVDiff/BEVDet

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ── CPU thread budget ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # projects/ and mmdet3d/ resolve to BEVDet

GPUS=${GPUS:-4}
PORT=${PORT:-29517}

RECIPE=${RECIPE:-}
case "$RECIPE" in
    nocbgs) C0_CKPT="../results/bevdet/det/bevdet4d-r50-nocbgs_seed0/epoch_24_ema.pth" ;;
    cbgs)   C0_CKPT="../results/bevdet/det/bevdet4d-r50-cbgs_seed0/epoch_20_ema.pth" ;;
    *) echo "error: set RECIPE=nocbgs or RECIPE=cbgs (C0 recipe of the frozen target), got '$RECIPE'" >&2; exit 1 ;;
esac

BEV_CONFIG="projects/configs/bevdiffuser/bevdet4d_r50_onlyseg_sam_v2_metric3d.py"
BEV_CHECKPOINT=${BEV_CHECKPOINT:-"$C0_CKPT"}
PRETRAINED_MODEL="projects/bevdiffuser/sd21_scheduler"   # local SD-2.1 scheduler config (see its README)

# set up tracker
PROJ_NAME=BEVDiffuser
RUN_NAME=${RUN_NAME:-"BEVDiffuser_bevdet4d_r50_${RECIPE}_onlyseg_sam3_metric3d"}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-10000}
CHECKPOINT_LIMIT=3

MAX_TRAINING_STEPS=${MAX_TRAINING_STEPS:-50000}
TRAIN_BATCH_SIZE=4
DATALOADER_NUM_WORKERS=8
GRADIENT_ACCUMMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"

UNCOND_PROB=0.2   # seg CFG dropout
PREDICTION_TYPE="sample"
TASK_LOSS_SCALE=0
SEED=0

OUTPUT_DIR=${OUTPUT_DIR:-"../results/bevdiffuser/det/${RUN_NAME}"}
RESUME_FROM=${RESUME_FROM:-}
DRY_RUN=${DRY_RUN:-0}

NVIS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if [ "$NVIS" -ne "$GPUS" ]; then
    echo "error: GPUS=$GPUS but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has $NVIS devices" >&2; exit 1
fi
if [ "$GPUS" -ne 4 ]; then
    echo "warning: GPUS=$GPUS gives batch $((GPUS * TRAIN_BATCH_SIZE)), not the flagship 16 (smoke tests only)" >&2
fi
[ -f "$BEV_CHECKPOINT" ] || { echo "error: missing BEV_CHECKPOINT $BEV_CHECKPOINT (train C0 first)" >&2; exit 1; }

python -c "import mmdet3d, sys; sys.exit(0 if mmdet3d.__version__.startswith('1.0.0') else 1)" || {
    echo "error: activate jhl-bevdet (BEVDet mmdet3d 1.0.0rc4 fork) first" >&2; exit 1; }

EXTRA=()
if [ -n "$RESUME_FROM" ]; then
    [ -d "$RESUME_FROM" ] || { echo "error: missing RESUME_FROM $RESUME_FROM" >&2; exit 1; }
    EXTRA+=(--resume_from_checkpoint "$RESUME_FROM")
elif compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
    echo "error: $OUTPUT_DIR already has checkpoints; set RESUME_FROM or a new RUN_NAME" >&2; exit 1
fi

CMD=(torchrun --nproc_per_node "$GPUS" --master_port="$PORT"
  projects/bevdiffuser/train_bev_diffuser_only_seg_bevdet.py
    --bev_config "$BEV_CONFIG"
    --bev_checkpoint "$BEV_CHECKPOINT"
    --pretrained_model_name_or_path "$PRETRAINED_MODEL"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --gradient_accumulation_steps "$GRADIENT_ACCUMMULATION_STEPS"
    --max_train_steps "$MAX_TRAINING_STEPS"
    --learning_rate "$LEARNING_RATE"
    --lr_scheduler "$LR_SCHEDULER"
    --output_dir "$OUTPUT_DIR"
    --checkpoints_total_limit "$CHECKPOINT_LIMIT"
    --checkpointing_steps "$CHECKPOINT_STEP"
    --tracker_run_name "$RUN_NAME"
    --tracker_project_name "$PROJ_NAME"
    --uncond_prob "$UNCOND_PROB"
    --prediction_type "$PREDICTION_TYPE"
    --task_loss_scale "$TASK_LOSS_SCALE"
    --seed "$SEED"
    --report_to 'tensorboard'
    "${EXTRA[@]}" "$@")

if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
{
    echo "date: $(date '+%F %T %Z')  host: $(hostname)  python: $(command -v python)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  RECIPE=$RECIPE"
    echo "cmd: $(printf '%q ' "${CMD[@]}")"
} > "$OUTPUT_DIR/launch_$(date +%Y%m%d_%H%M%S).txt"

"${CMD[@]}"
