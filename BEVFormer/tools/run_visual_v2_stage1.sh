#!/usr/bin/env bash
# Run baseline-vs-ours detection visualization (tools/analysis_tools/visual_v2.py).
#
# Pipeline:
#   1) (optional) run tools/test.py with --eval bbox for each model to produce
#      results_nusc.json with a fixed jsonfile_prefix
#   2) run tools/analysis_tools/visual_v2.py to render per-sample camera/BEV
#      comparison figures
#
# Usage:
#   bash tools/run_visual_v2.sh           # full pipeline (test + visualize)
#   SKIP_TEST=1 bash tools/run_visual_v2.sh   # skip step 1, only visualize
#
# Run from the BEVFormer/ directory (same as dist_train_seg.sh).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

GPUS=${GPUS:-4}
PORT_OURS=${PORT_OURS:-29505}
PORT_BASE=${PORT_BASE:-29506}
# Existing JSONs are already produced under <model_dir>/val/<timestamp>/pts_bbox/.
# Default to SKIP_TEST=1 to reuse the latest one; set SKIP_TEST=0 to re-run test.py.
SKIP_TEST=${SKIP_TEST:-1}

# ---------- model paths ----------
OURS_DIR="../results/version2/stage2/DiffBEVFormer_tiny_seg_v5"
OURS_CONFIG="./projects/configs/diff_bevformer/layout_tiny_seg_v4_adapter.py"
OURS_CKPT="${OURS_DIR}/epoch_24.pth"

BASE_DIR="../results/version2/stage2/DiffBEVFormer_tiny_original_24epoch"
BASE_CONFIG="./projects/configs/diff_bevformer/layout_tiny.py"
BASE_CKPT="${BASE_DIR}/epoch_24.pth"

# ---------- JSON paths ----------
OURS_JSON="./test/layout_tiny_seg_v4/BEVDiffuser_tiny_seg_one-hot_v11/checkpoint-50000/1001_1001_50/pts_bbox/results_nusc.json"
BASE_JSON="./test/layout_tiny/BEVDiffuser_BEVFormer_tiny_original_bs2/checkpoint-50000/1001_1001_50/pts_bbox/results_nusc.json"

# When SKIP_TEST=0, force a re-run with a fixed jsonfile_prefix so we know the path.
OURS_JSON_PREFIX="${OURS_DIR}/val/run_visual_v2"
BASE_JSON_PREFIX="${BASE_DIR}/val/run_visual_v2"

VIS_OUT_DIR=${VIS_OUT_DIR:-"../results/visualize_stage1/det_vis/BEV_generation"}

# ---------- visualization params ----------
DATA_ROOT="./data/nuscenes"
NUSC_VERSION="v1.0-trainval"
START_IDX=0
NUM_SAMPLES=20
SCORE_THR=0.3
DPI=300
# Optional scene_token whitelist. If non-empty, ONLY samples that belong to
# these scenes are visualized and SCENE_KEYWORDS is ignored.
# e.g. SCENE_TOKENS=("fcbccedd61424f1b85dcbf8f897f9754")
SCENE_TOKENS=()
# Optional scene-description filter, e.g. ("rain") or ("rain" "night").
# Leave empty to disable. SCENE_MATCH ∈ {any, all}.
# Ignored when SCENE_TOKENS is non-empty.
# SCENE_KEYWORDS=("rain" "night")
SCENE_KEYWORDS=("rain")
SCENE_MATCH="any"

mkdir -p "${VIS_OUT_DIR}"

run_test () {
    local CFG=$1
    local CKPT=$2
    local PORT=$3
    local PREFIX=$4

    echo "[run_visual_v2] testing  CFG=${CFG}"
    echo "[run_visual_v2]          CKPT=${CKPT}"
    echo "[run_visual_v2]          jsonfile_prefix=${PREFIX}"

    PYTHONPATH="$(dirname $0)/..":${PYTHONPATH:-} \
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        $(dirname "$0")/test.py "${CFG}" "${CKPT}" \
        --launcher pytorch \
        --eval bbox \
        --eval-options jsonfile_prefix="${PREFIX}"
}

if [ "${SKIP_TEST}" != "1" ]; then
    run_test "${OURS_CONFIG}" "${OURS_CKPT}" "${PORT_OURS}" "${OURS_JSON_PREFIX}"
    run_test "${BASE_CONFIG}" "${BASE_CKPT}" "${PORT_BASE}" "${BASE_JSON_PREFIX}"
    OURS_JSON="${OURS_JSON_PREFIX}/pts_bbox/results_nusc.json"
    BASE_JSON="${BASE_JSON_PREFIX}/pts_bbox/results_nusc.json"
else
    echo "[run_visual_v2] SKIP_TEST=1 → reusing existing JSONs"
fi

if [ -z "${OURS_JSON}" ] || [ ! -f "${OURS_JSON}" ]; then
    echo "[run_visual_v2] ERROR: ours JSON not found (looked under ${OURS_DIR}/val/*/pts_bbox/results_nusc.json)" >&2
    exit 1
fi
if [ -z "${BASE_JSON}" ] || [ ! -f "${BASE_JSON}" ]; then
    echo "[run_visual_v2] ERROR: baseline JSON not found (looked under ${BASE_DIR}/val/*/pts_bbox/results_nusc.json)" >&2
    exit 1
fi
echo "[run_visual_v2] ours JSON     : ${OURS_JSON}"
echo "[run_visual_v2] baseline JSON : ${BASE_JSON}"

SCENE_ARGS=()
if [ ${#SCENE_TOKENS[@]} -gt 0 ]; then
    SCENE_ARGS+=(--scene_tokens "${SCENE_TOKENS[@]}")
    echo "[run_visual_v2] scene_token filter: ${SCENE_TOKENS[*]}"
elif [ ${#SCENE_KEYWORDS[@]} -gt 0 ]; then
    SCENE_ARGS+=(--scene_keywords "${SCENE_KEYWORDS[@]}" --scene_match "${SCENE_MATCH}")
    echo "[run_visual_v2] scene filter [${SCENE_MATCH}]: ${SCENE_KEYWORDS[*]}"
fi

echo "[run_visual_v2] visualizing → ${VIS_OUT_DIR}"
PYTHONPATH="$(dirname $0)/..":${PYTHONPATH:-} \
python $(dirname "$0")/analysis_tools/visual_v2.py \
    --results_baseline "${BASE_JSON}" \
    --results_ours    "${OURS_JSON}" \
    --data_root       "${DATA_ROOT}" \
    --version         "${NUSC_VERSION}" \
    --start_idx       "${START_IDX}" \
    --num_samples     "${NUM_SAMPLES}" \
    --score_thr       "${SCORE_THR}" \
    --dpi             "${DPI}" \
    --out_dir         "${VIS_OUT_DIR}" \
    "${SCENE_ARGS[@]}"

echo "[run_visual_v2] done. figures at ${VIS_OUT_DIR}"
