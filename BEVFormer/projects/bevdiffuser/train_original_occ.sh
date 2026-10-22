#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BEVFORMER_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# nuScenes info files contain image paths relative to BEVFormer/.
cd "$BEVFORMER_DIR"

# export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPUS=8
PORT=${PORT:-29515}

BEV_CONFIG="$BEVFORMER_DIR/projects/configs/bevdiffuser/layout_base_occ.py"
BEV_CHECKPOINT="$BEVFORMER_DIR/../results/occ3d/bevformer_base_occ/epoch_24.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

# set up wandb project
PROJ_NAME=BEVDiffuser-Occ3D
RUN_NAME=BEVDiffuser_layout_base_occ

# checkpoint settings
CHECKPOINT_STEP=10000
CHECKPOINT_LIMIT=3

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=4
DATALOADER_NUM_WORKERS=8
GRADIENT_ACCUMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4
LR_SCHEDULER="constant" # cyclic, cosine, constant

UNCOND_PROB=0.2
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0.1 # 0.1

OUTPUT_DIR="$BEVFORMER_DIR/../results/bevdiffuser/${RUN_NAME}"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$PRETRAINED_UNET_CHECKPOINT" != "None" ]]; then
    EXTRA_ARGS+=(--pretrained_unet_checkpoint "$PRETRAINED_UNET_CHECKPOINT")
fi

# train!
# taskset -c 16-31,48-63 torchrun \
PYTHONPATH="$BEVFORMER_DIR${PYTHONPATH:+:$PYTHONPATH}" \
torchrun --nproc_per_node "$GPUS" \
    --master_port "$PORT" \
  "$SCRIPT_DIR/train_bev_diffuser_occ.py" \
    --bev_config "$BEV_CONFIG" \
    --bev_checkpoint "$BEV_CHECKPOINT" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_train_steps $MAX_TRAINING_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULER \
    --output_dir "$OUTPUT_DIR" \
    --checkpoints_total_limit $CHECKPOINT_LIMIT \
    --checkpointing_steps $CHECKPOINT_STEP \
    --tracker_run_name "$RUN_NAME" \
    --tracker_project_name "$PROJ_NAME" \
    --uncond_prob $UNCOND_PROB \
    --prediction_type $PREDICTION_TYPE \
    --task_loss_scale $TASK_LOSS_SCALE \
    --report_to 'tensorboard' \
    "${EXTRA_ARGS[@]}"
