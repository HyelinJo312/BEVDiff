set -e

export CUDA_VISIBLE_DEVICES=1

BEV_CONFIG="../configs/bevdiffuser/layout_tiny_dino.py"

CHECKPOINT_DIR="../../../results/pretrain_stage1/BEVDiffuser_BEVFormer_tiny_dino_v2_new_global_adapter_constant-lr/checkpoint-60000"

BEV_CHECKPOINT="../../../results/pretrain_stage1/BEVDiffuser_BEVFormer_tiny_dino_v2_new_global_adapter_constant-lr/checkpoint-60000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1 
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1


python -m torch.distributed.launch --master_port 9997 test_bev_diffuser_dino_v2_vis.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 0 \
    --denoise_timesteps 0 \
    --num_inference_steps 0 \
    # --use_classifier_guidence \


