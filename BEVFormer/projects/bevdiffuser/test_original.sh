set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"
# BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"

CHECKPOINT_DIR="../../../results/BEVDiffuser_BEVFormer_tiny_original_bs2/checkpoint-50000"

BEV_CHECKPOINT="../../../results/BEVDiffuser_BEVFormer_tiny_original_bs2/checkpoint-50000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

# export NCCL_SOCKET_IFNAME=lo
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1 
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1


# python -m torch.distributed.launch --master_port 9995 test_bev_diffuser_dino.py \
torchrun --nproc_per_node=4 \
    --master_port 9993 \
    test_bev_diffuser.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 1001 \
    --denoise_timesteps 1001 \
    --num_inference_steps 50 \
    # --use_classifier_guidence \


