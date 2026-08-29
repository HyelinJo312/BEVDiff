#!/usr/bin/env bash
# Run Experiment 4 global feature alignment analysis without region-wise
# metrics or visualization.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

NUM_FRAMES=${NUM_FRAMES:-200}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
OUTPUT_DIR=${OUTPUT_DIR:-../results/feature_alignment/exp4_global_no_region}
TEACHER_INIT=${TEACHER_INIT:-./ckpts/bevformer_tiny_epoch_24.pth}
PYTHON=${PYTHON:-python}

BASELINE_DIR="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch"
SEM_MSE_DIR="../results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_no-mgd_t100"
SEM_MGD_DIR="../results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_mgd_v4_2_t100_alpha100_lambda_0.6"

"${PYTHON}" tools/analysis_tools/analyze_feature_alignment.py \
    --baseline_config "${BASELINE_DIR}/layout_tiny.py" \
    --baseline_ckpt "${BASELINE_DIR}/epoch_24.pth" \
    --semantic_mse_config "${SEM_MSE_DIR}/bev_tiny_onlyseg_sam_v2.py" \
    --semantic_mse_ckpt "${SEM_MSE_DIR}/epoch_24.pth" \
    --semantic_mgd_config "${SEM_MGD_DIR}/bev_tiny_onlyseg_sam_v2.py" \
    --semantic_mgd_ckpt "${SEM_MGD_DIR}/epoch_24.pth" \
    --teacher_init_ckpt "${TEACHER_INIT}" \
    --num_frames "${NUM_FRAMES}" \
    --workers_per_gpu "${WORKERS_PER_GPU}" \
    --output_dir "${OUTPUT_DIR}"
