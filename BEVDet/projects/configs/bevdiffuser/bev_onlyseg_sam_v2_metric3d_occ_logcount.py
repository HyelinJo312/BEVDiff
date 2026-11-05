# Stage 1 occupancy teacher with a share + log-count semantic prior.
# Identical to bev_onlyseg_sam_v2_metric3d_occ.py (the v3 teacher) except the
# semantic input: [share (17) | log1p(per-class pixel count) / log1p(1e5) (17)].
# Offline check (results/analysis/prior_variants): log-count raises small-object
# prior AP ~2x over share (cone 0.139 -> 0.348, pedestrian 0.164 -> 0.472).
#
# Build the cache first (from BEVFormer/projects/bevdiffuser):
#   CONFIG=../configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ_logcount.py \
#   CACHE_ROOT=../../data/semantic_bev_occ_v2_counts SOURCE_VERSION=semantic_bev_occ_ego \
#   bash build_semantic_bev_cache.sh
_base_ = ['./bev_onlyseg_sam_v2_metric3d_occ.py']

semantic_type = 'share+logcount'
semantic_logcount_scale = 1e5

# Stored cache layout read by build_semantic_bev_cache.py: share | counts | support.
semantic_bev_cache_layout = 'share+counts+support'
semantic_bev_cache_root = '../../data/semantic_bev_occ_v2_counts'

unet = dict(parameters=dict(
                seg_bev_aligner=dict(
                    semantic_type=semantic_type, 
                    logcount_scale=semantic_logcount_scale))
)

_semantic = dict(semantic_bev_cache_root=semantic_bev_cache_root,
                 semantic_type=semantic_type,
                 semantic_logcount_scale=semantic_logcount_scale)

occ_conditions = dict(**_semantic)
data = dict(train=dict(**_semantic), 
            val=dict(**_semantic), 
            test=dict(**_semantic))
