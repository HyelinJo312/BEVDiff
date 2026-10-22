#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG=${CONFIG:-../configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py}
CACHE_ROOT=${CACHE_ROOT:-../../data/semantic_bev_occ_v2}
SOURCE_VERSION=${SOURCE_VERSION:-semantic_bev_occ_ego}
SPLIT=${SPLIT:-all}
FLUSH_EVERY=${FLUSH_EVERY:-50}
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    printf 'CUDA_VISIBLE_DEVICES must be a comma-separated list of physical GPU indices\n' >&2
    exit 2
fi
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
devices=()
declare -A seen_devices=()
for index in "${!visible_devices[@]}"; do
    physical_device=${visible_devices[$index]}
    if [[ -n ${seen_devices[$physical_device]+set} ]]; then
        printf 'Duplicate physical GPU in CUDA_VISIBLE_DEVICES: %s\n' "$physical_device" >&2
        exit 2
    fi
    seen_devices[$physical_device]=1
    devices+=("cuda:$index")
done
printf 'Cache root: %s\nVisible GPUs: %s\nLogical devices: %s\n' "$CACHE_ROOT" "$CUDA_VISIBLE_DEVICES" "${devices[*]}"

"$PYTHON" build_semantic_bev_cache.py \
    --config "$CONFIG" \
    --cache-root "$CACHE_ROOT" \
    --source-version "$SOURCE_VERSION" \
    --split "$SPLIT" \
    --devices "${devices[@]}" \
    --flush-every "$FLUSH_EVERY" \
    "$@"
