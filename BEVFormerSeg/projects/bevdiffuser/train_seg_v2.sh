#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

# ── CPU thread budget (NUMA node 0: CPU 0-15, 32-47 = 32 logical cores) ─────
# taskset 고정 범위 내: (1 main + 4 workers) × OMP=4 = 20 threads → 여유 있음
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1     # prevent OpenBLAS from spawning extra threads
export OPENCV_NUM_THREADS=1       # 1=single thread (0=use all cores → thread explosion in dataloader workers)
export MAX_JOBS=16                 # parallel build jobs (half server)
export TCNN_CUDA_ARCHITECTURES=86 # Ampere (A6000)

# ── NCCL / DDP stability ─────────────────────────────────────────────────────
# export NCCL_IB_DISABLE=1             # no InfiniBand on this server
# export NCCL_P2P_DISABLE=0            # GPU 0-3: 같은 NUMA node → NODE 토폴로지 PCIe P2P 활용


GPUS=4
PORT=${PORT:-29501}   # train_seg_v2 전용 포트 (v3는 29503)

BEV_CONFIG="../configs/bevdiffuser/layout_tiny_seg_det_semantic_v4_2.py"
BEV_CHECKPOINT="../../results/bevformer_tiny_seg_v3/epoch_24.pth"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

# set up wandb project
PROJ_NAME=BEVDiffuser
RUN_NAME=BEVDiffuserSeg_tiny_v3_task-loss_01_attn421_segmap2
# checkpoint settings
CHECKPOINT_STEP=10000
CHECKPOINT_LIMIT=3

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=2
DATALOADER_NUM_WORKERS=4
GRADIENT_ACCUMMULATION_STEPS=1

# loss and lr settings
LEARNING_RATE=1e-4
LR_SCHEDULER="constant" # constant, constant_with_warmup, polynomial, cosine_with_restarts

UNCOND_PROB=0.2   # 0.2 -> 0.1
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=0.1 # 0.1

OUTPUT_DIR="../../results/bevdiffuser/stage1/${RUN_NAME}"

# RESUME_FROM="../../results/bevdiffuser/stage1/BEVDiffuserSeg_tiny_v3_da3_task-loss_01/checkpoint-40000"

mkdir -p $OUTPUT_DIR


# train!
export PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH
# taskset: NUMA node 1 (CPU 16-31, 48-63) 고정 → GPU 4-7와 NUMA-local
# taskset -c 16-31,48-63 torchrun --nproc_per_node $GPUS \
torchrun --nproc_per_node $GPUS \
    --master_port=$PORT \
  "$(dirname "$0")/train_bev_diffuser_seg_v2.py" \
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
    # --resume_from_checkpoint $RESUME_FROM
    # --gradient_checkpointing

