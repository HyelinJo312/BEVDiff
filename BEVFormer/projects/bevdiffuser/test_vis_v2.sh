set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

BEV_CONFIG="../configs/bevdiffuser/layout_tiny_seg_v4.py"

CHECKPOINT_DIR="../../../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000"

BEV_CHECKPOINT="../../../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1 
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1

# python -m torch.distributed.launch --master_port 9995 test_bev_diffuser_dino_vis.py \
torchrun --nproc_per_node=4 \
    --master_port 9995 \
    test_bev_diffuser_seg_vis.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 100 \
    --denoise_timesteps 100 \
    --num_inference_steps 5 \
    # --use_classifier_guidence \


