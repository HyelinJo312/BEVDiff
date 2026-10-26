set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

BEV_CONFIG="../configs/bevdiffuser/bev_tiny_det_semantic_onlyseg_sam_metric3d.py"
# BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"

CHECKPOINT_DIR="../../../results/map_seg/bevdiffuser/BEVDiffuserSeg_tiny_onlyseg_sam3_metric3d/checkpoint-50000"

BEV_CHECKPOINT="../../../results/map_seg/bevdiffuser/BEVDiffuserSeg_tiny_onlyseg_sam3_metric3d/checkpoint-50000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"


# python -m torch.distributed.launch --master_port 9995 test_bev_diffuser_dino.py \
torchrun --nproc_per_node=4 \
    --master_port 9997 \
    test_bev_diffuser_only_seg.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 100 \
    --denoise_timesteps 100 \
    --num_inference_steps 5 \
    # --use_classifier_guidence \


