#!/usr/bin/env bash
# Run from BEVFormer/projects/bevdiffuser/. Cache settings come from BEV_CONFIG.

# GPU selection: using GPUs 4-7 (0-3 occupied by other jobs)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ── CPU thread budget ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no


GPUS=4
PORT=${PORT:-29508}

BEV_CONFIG="../configs/bevdiffuser/bev_base_onlyseg_sam_v2_metric3d_v3.py"
BEV_CHECKPOINT="../../ckpts/bevformer_r101_dcn_24ep.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT="../../../results/bevdiffuser/det/BEVDiffuser_tiny_onlyseg_sam3_metric3d_v3_scale-shift/checkpoint-50000"

# set up wandb project
PROJ_NAME=BEVDiffuser
RUN_NAME=BEVDiffuser_base_onlyseg_sam3_metric3d_v3_scale-shift
CHECKPOINT_STEP=10000
CHECKPOINT_LIMIT=3

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=4
DATALOADER_NUM_WORKERS=8
GRADIENT_ACCUMMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4
LR_SCHEDULER="constant" # constant, constant_with_warmup, polynomial, cosine_with_restarts

UNCOND_PROB=0.2   # seg CFG dropout
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0

OUTPUT_DIR="../../../results/bevdiffuser/det/${RUN_NAME}"
# RESUME_FROM="../../../results/version2/stage1/BEVDiffuser_tiny_onlyseg_sam3/checkpoint-40000"

mkdir -p "$OUTPUT_DIR"


# train!
export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"
# taskset -c 0-15,32-47 torchrun --nproc_per_node $GPUS \
torchrun --nproc_per_node $GPUS \
    --master_port=$PORT \
  train_bev_diffuser_only_seg.py \
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
    --report_to 'tensorboard'

    # --resume_from_checkpoint $RESUME_FROM
    # --gradient_checkpointing
