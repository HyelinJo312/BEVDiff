#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEVFORMER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$BEVFORMER_DIR/.." && pwd)"

# GPU selection: using GPUs 4-7 (0-3 occupied by other jobs)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ── CPU thread budget ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_NUM_THREADS=1       # 1=단일 스레드 유지 (스레드 폭발로 인한 극심한 지연 방지)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

# ── NCCL / inter-GPU communication ──────────────────────────────────────────
# export NCCL_IB_DISABLE=1          # no InfiniBand on this server
# export NCCL_P2P_DISABLE=0         # enable PCIe/NVLink P2P between GPUs


GPUS=4
PORT=${PORT:-29506}

BEV_CONFIG="$BEVFORMER_DIR/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py"
# Omit overrides to use the config; 0=live SAM3/depth, 1=local probabilities.
CACHE_CFG_OPTIONS=()
case "${USE_SEMANTIC_BEV_CACHE:-config}" in
    config) ;;
    0|1)
        CACHE_ENABLED=False
        [[ "$USE_SEMANTIC_BEV_CACHE" == 1 ]] && CACHE_ENABLED=True
        for SPLIT in train val test; do
            CACHE_CFG_OPTIONS+=("data.$SPLIT.use_semantic_bev_cache=$CACHE_ENABLED")
        done
        ;;
    *) echo "USE_SEMANTIC_BEV_CACHE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${SEMANTIC_BEV_CACHE_ROOT:-}" ]]; then
    for SPLIT in train val test; do
        CACHE_CFG_OPTIONS+=("data.$SPLIT.semantic_bev_cache_root=$SEMANTIC_BEV_CACHE_ROOT")
    done
fi
CACHE_CFG_ARGS=()
if (( ${#CACHE_CFG_OPTIONS[@]} )); then
    CACHE_CFG_ARGS=(--cfg-options "${CACHE_CFG_OPTIONS[@]}")
fi
BEV_CHECKPOINT="$BEVFORMER_DIR/ckpts/bevformer_tiny_epoch_24.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

# set up wandb project
PROJ_NAME=BEVDiffuser
RUN_NAME=BEVDiffuser_tiny_onlyseg_sam3_v7
CHECKPOINT_STEP=10000
CHECKPOINT_LIMIT=3

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=2
DATALOADER_NUM_WORKERS=8
GRADIENT_ACCUMMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4
LR_SCHEDULER="constant" # constant, constant_with_warmup, polynomial, cosine_with_restarts

UNCOND_PROB=0.2   # seg CFG dropout
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0 # 0: dinoseg 대비 동일 조건 ablation. task-aware 실험 시 0.1

OUTPUT_DIR="$WORKSPACE_DIR/results/version2/stage1/${RUN_NAME}"
# RESUME_FROM="$WORKSPACE_DIR/results/version2/stage1/BEVDiffuser_tiny_onlyseg_sam3/checkpoint-40000"

mkdir -p "$OUTPUT_DIR"


# train!
export PYTHONPATH="$BEVFORMER_DIR":$PYTHONPATH
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
    echo "Error: torchrun not found. Activate the training environment first, e.g. 'conda activate jhl-bevdiffuser'." >&2
    exit 127
fi
# taskset -c 0-15,32-47 torchrun --nproc_per_node $GPUS \
# torchrun --nproc_per_node $GPUS \
taskset -c 16-31,48-63 "$TORCHRUN_BIN" --nproc_per_node $GPUS \
    --master_port=$PORT \
  "$SCRIPT_DIR/train_bev_diffuser_only_seg.py" \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --pretrained_unet_checkpoint $PRETRAINED_UNET_CHECKPOINT \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --gradient_accumulation_steps $GRADIENT_ACCUMMULATION_STEPS \
    --max_train_steps $MAX_TRAINING_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULER \
    --output_dir $OUTPUT_DIR \
    --checkpoints_total_limit $CHECKPOINT_LIMIT \
    --checkpointing_steps $CHECKPOINT_STEP \
    --tracker_run_name $RUN_NAME \
    --tracker_project_name $PROJ_NAME \
    --uncond_prob $UNCOND_PROB \
    --prediction_type $PREDICTION_TYPE \
    --task_loss_scale $TASK_LOSS_SCALE \
    --report_to 'tensorboard' \
    "${CACHE_CFG_ARGS[@]}" \
    # --resume_from_checkpoint $RESUME_FROM
    # --gradient_checkpointing
