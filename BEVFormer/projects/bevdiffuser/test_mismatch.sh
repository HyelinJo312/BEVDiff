set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

BEV_CONFIG="../configs/bevdiffuser/layout_tiny_seg_v4.py"

CHECKPOINT_DIR="../../../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000"

BEV_CHECKPOINT="../../../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000/bev_model.pth"

PREDICTION_TYPE="sample"

NOISE=5
DENOISE=5
STEPS=5
DONOR_OFFSET=500

# Run all three modes for a fair within-script comparison.
#   matched : image = condition = anchor                       (control)
#   cond_gt : image = donor(B), condition = anchor(A=GT)       -> A-GT stays high => condition shortcut
#   img_gt  : image = anchor(B=GT), condition = donor(A)       -> B-GT stays high => image dependence
for MODE in matched cond_gt img_gt; do
    echo "==================== mismatch_mode = ${MODE} ===================="
    torchrun --nproc_per_node=4 \
        --master_port 9993 \
        test_bev_diffuser_seg_mismatch.py \
        --bev_config $BEV_CONFIG \
        --bev_checkpoint $BEV_CHECKPOINT \
        --checkpoint_dir $CHECKPOINT_DIR \
        --prediction_type $PREDICTION_TYPE \
        --noise_timesteps $NOISE \
        --denoise_timesteps $DENOISE \
        --num_inference_steps $STEPS \
        --mismatch_mode $MODE \
        --donor_offset $DONOR_OFFSET \
        --disable_temporal True \
        --eval bbox
done
