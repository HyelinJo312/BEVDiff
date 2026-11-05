set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

BEV_CONFIG="../configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ_v2.py"
# BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"

CHECKPOINT_DIR="../../../results/bevdiffuser/BEVDiffuser_occ_onlyseg_sam3_metric3d_v4/checkpoint-20000"

# BEV_CHECKPOINT="../../../results/bevdiffuser/BEVDiffuser_tiny_onlyseg_sam3_metric3d/checkpoint-50000/bev_model.pth"
BEV_CHECKPOINT="../../../results/diff_bevformer/occ/baseline/bevformer_base_occ/epoch_24.pth"
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
    --master_port 9998 \
    test_bev_diffuser_occ_only_seg.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 300 \
    --denoise_timesteps 300 \
    --num_inference_steps 5 \
    --guidance_scale 4.0
    # --use_classifier_guidence \


