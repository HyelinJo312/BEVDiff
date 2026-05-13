set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- Two BEV configs (baseline + ours) ----
BEV_CONFIG_BASELINE="../configs/bevdiffuser/layout_tiny.py"
BEV_CONFIG_OURS="../configs/bevdiffuser/layout_tiny_seg_v4.py"

# ---- Two UNet checkpoint dirs ----
CHECKPOINT_DIR_BASELINE="../../../results/BEVDiffuser_BEVFormer_tiny_original_bs2/checkpoint-50000"
CHECKPOINT_DIR_OURS="../../../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000"

# ---- Shared BEVFormer detector checkpoint ----
BEV_CHECKPOINT="../../../results/version2/stage1/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000/bev_model.pth"

PREDICTION_TYPE="sample"

export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1

# Independent sampling hyperparameters per model
# (baseline uses BEVDiffuser dynamics; ours typically benefits from the same or
# different noise/denoise schedule — tune freely without affecting the other model.)
torchrun --nproc_per_node=4 \
    --master_port 9995 \
    test_bev_diffuser_seg_vis_v2.py \
    --bev_config_baseline   $BEV_CONFIG_BASELINE \
    --bev_config_ours       $BEV_CONFIG_OURS \
    --bev_checkpoint        $BEV_CHECKPOINT \
    --checkpoint_dir_baseline $CHECKPOINT_DIR_BASELINE \
    --checkpoint_dir_ours     $CHECKPOINT_DIR_OURS \
    --prediction_type       $PREDICTION_TYPE \
    --baseline_noise_t      5 \
    --baseline_denoise_t    5 \
    --baseline_inference_steps 5 \
    --ours_noise_t          100 \
    --ours_denoise_t        100 \
    --ours_inference_steps  5 \
    --cfg_scale             2.0 \
    --vis_layout            1x4 \
    --vis_mode              activation \
    --out_subdir            bev_compare \
    # --vis_surround \
    # --surround_width        2400 \
    # --use_classifier_guidence \
