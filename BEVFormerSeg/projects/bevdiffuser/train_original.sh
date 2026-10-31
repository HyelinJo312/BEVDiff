#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1

# ── CPU thread budget ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELERATE_MIXED_PRECISION=no

GPUS=4
PORT=${PORT:-29503}

BEV_CONFIG="../configs/bevdiffuser/layout_tiny_mapseg.py"
BEV_CHECKPOINT="../../../results/map_seg/baseline/bevformer_tiny_seg_v3/epoch_24.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

# set up wandb project
PROJ_NAME=BEVDiffuserSeg
RUN_NAME=BEVDiffuserSeg_tiny_original

# checkpoint settings
CHECKPOINT_STEP=10000
CHECKPOINT_LIMIT=3

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=4
DATALOADER_NUM_WORKERS=8
GRADIENT_ACCUMMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4  
LR_SCHEDULER="constant" # cyclic, cosine, constant

UNCOND_PROB=0.2
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0.1 # seg head loss, as BEVDiffuser uses its task head

OUTPUT_DIR="../../../results/map_seg/bevdiffuser/${RUN_NAME}"


mkdir -p $OUTPUT_DIR


# train!
export PYTHONPATH="$(dirname $0)/../..${PYTHONPATH:+:$PYTHONPATH}"
taskset -c 16-31,48-63 torchrun \
    --nproc_per_node $GPUS \
    --master_port=$PORT \
  $(dirname "$0")/train_bev_diffuser.py \
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
    # --gradient_checkpointing


