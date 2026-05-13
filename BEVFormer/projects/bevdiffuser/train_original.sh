#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_NUM_THREADS=1       # let OpenCV auto-detect (0 = use OMP setting)
export MAX_JOBS=8                 # 빌드 job 상한 (과도한 fork 방지)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

# ── NCCL / inter-GPU communication ──────────────────────────────────────────
export NCCL_IB_DISABLE=1          # no InfiniBand on this server
# export NCCL_P2P_DISABLE=0         # GPU 0-3: 같은 NUMA node → NODE 토폴로지 P2P 활용
# export NCCL_SOCKET_NTHREADS=4     # NUMA 격리 후 소켓 스레드 복원
# export NCCL_NSOCKS_PERTHREAD=2    # 스레드당 소켓 수

GPUS=4
PORT=${PORT:-29503}

BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"
BEV_CHECKPOINT="../../ckpts/bevformer_tiny_epoch_24.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

# set up wandb project
PROJ_NAME=BEVDiffuser
RUN_NAME=BEVDiffuser_BEVFormer_tiny_original_bs4

# checkpoint settings
CHECKPOINT_STEP=10000
CHECKPOINT_LIMIT=5

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=4
DATALOADER_NUM_WORKERS=4
GRADIENT_ACCUMMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4  
LR_SCHEDULER="constant" # cyclic, cosine, constant

UNCOND_PROB=0.2
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0.1 # 0.1

OUTPUT_DIR="../../../results/${RUN_NAME}"

mkdir -p $OUTPUT_DIR

# train!
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
# taskset -c 16-31,48-63 torchrun \
torchrun --nproc_per_node $GPUS \
    --nproc_per_node $GPUS \
    --master_port=29505 \
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
    --report_to 'tensorboard' \
    # --gradient_checkpointing \


