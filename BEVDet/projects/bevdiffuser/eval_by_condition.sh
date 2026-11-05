set -e

export CUDA_VISIBLE_DEVICES=0

BEV_CONFIG="../configs/diff_bevformer/layout_tiny_seg_v4_2.py"

# Point this at the results_nusc.json produced by a previous test.sh run.
RESULT_PATH="../../test/layout_tiny_seg_v4_2/BEVDiffuser_tiny_seg_one-hot_ablation_FDN/checkpoint-50000/5_5_5/pts_bbox/results_nusc.json"

python eval_by_condition.py \
    --bev_config $BEV_CONFIG \
    --result_path $RESULT_PATH
