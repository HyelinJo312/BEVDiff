#!/usr/bin/env bash

# GPU selection: using GPUs 4-7 (0-3 occupied by other jobs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
PORT=${PORT:-29501}

BEV_CONFIG="../configs/bevdiffuser/bev_tiny_dino_seg_sam.py"
BEV_CHECKPOINT="../../ckpts/bevformer_tiny_epoch_24.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

# set up wandb project
PROJ_NAME=BEVDiffuser
RUN_NAME=BEVDiffuser_tiny_dinoseg_v3_sam
# checkpoint settings
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

UNCOND_PROB=0.2   # 0.2 -> 0.1
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0 # 0.1

OUTPUT_DIR="../../../results/version2/stage1/${RUN_NAME}"
# RESUME_FROM="../../../results/stage1/BEVDiffuser_tiny_GT-dino_only-dino_with-global/checkpoint-30000"

mkdir -p $OUTPUT_DIR


# train!
# torchrun \
export PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH
taskset -c 0-15,32-47 torchrun --nproc_per_node $GPUS \
    --nproc_per_node $GPUS \
    --master_port=29505 \
  "$(dirname "$0")/train_bev_diffuser_dinoseg.py" \
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
    # --gradient_checkpointing
    # --resume_from_checkpoint $RESUME_FROM
    # --gradient_checkpointing


