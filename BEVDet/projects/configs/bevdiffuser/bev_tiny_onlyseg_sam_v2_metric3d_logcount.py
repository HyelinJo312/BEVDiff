# Stage 1 detection teacher with a share + log-count semantic prior.
# Identical to bev_tiny_onlyseg_sam_v2_metric3d.py except the semantic input:
# [share (17) | log1p(per-class pixel count) / log1p(1e5) (17)].
#
# Build the cache first (from BEVFormer/projects/bevdiffuser):
#   python build_semantic_bev_cache.py \
#     --config ../configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_logcount.py \
#     --cache-root ../../data/semantic_bev_counts --source-version sam3_metric3d_v1 \
#     --split all --devices cuda:0 ...
_base_ = ['./bev_tiny_onlyseg_sam_v2_metric3d.py']

semantic_type = 'share+logcount'
semantic_logcount_scale = 1e5

# Stored cache layout read by build_semantic_bev_cache.py: share | counts | support.
semantic_bev_cache_layout = 'share+counts+support'
semantic_bev_cache_root = '../../data/semantic_bev_counts'

unet = dict(parameters=dict(
                seg_bev_aligner=dict(
                    semantic_type=semantic_type, 
                    logcount_scale=semantic_logcount_scale))
)

_semantic = dict(semantic_bev_cache_root=semantic_bev_cache_root,
                 semantic_type=semantic_type,
                 semantic_logcount_scale=semantic_logcount_scale)

semantic_bev_cache_settings = dict(**_semantic)

data = dict(train=dict(**_semantic), 
            val=dict(**_semantic), 
            test=dict(**_semantic))
