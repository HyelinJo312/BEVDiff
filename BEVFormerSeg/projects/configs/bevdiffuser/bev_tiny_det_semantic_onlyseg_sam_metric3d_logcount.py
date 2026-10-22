# BEV segmentation teacher with a share + log-count semantic prior.
# Identical to bev_tiny_det_semantic_onlyseg_sam_metric3d.py except the semantic
# input: [share (18) | log1p(per-class pixel count) / log1p(1e5) (18)] = 36 channels.
# Mirrors BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_logcount.py.
#
# Taxonomy (seg_id_remap={17: 2, 18: 2, 19: 17}, inherited): crosswalk kept as class 16,
# lane/road arrow -> road(2), sky -> 17 = ignored id. num_classes=17 -> 18 share channels.
#
# Cache (lidar frame, +-51.2 m, 50x50, layout share+counts+support = 37 channels), built
# from BEVFormer/projects/bevdiffuser with
#   python build_semantic_bev_cache.py \
#     --config ../configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_logcount_crosswalk.py \
#     --cache-root /NHNHOME/data/hyelin/BEVDiff/BEVFormerSeg/data/semantic_bev_counts \
#     --source-version sam3_metric3d_v1 --split all --devices cuda:0 ...
_base_ = ['./bev_tiny_det_semantic_onlyseg_sam_metric3d.py']

semantic_type = 'share+logcount'
semantic_logcount_scale = 1e5

# Stored cache layout: share | counts | support.
semantic_bev_cache_layout = 'share+counts+support'
# Launchers run from the BEVFormerSeg repository root.
semantic_bev_cache_root = '../../data/semantic_bev_counts'

unet = dict(parameters=dict(
                seg_bev_aligner=dict(
                    num_classes=17,
                    semantic_type=semantic_type,
                    logcount_scale=semantic_logcount_scale))
)

_semantic = dict(semantic_bev_cache_root=semantic_bev_cache_root,
                 semantic_type=semantic_type,
                 semantic_logcount_scale=semantic_logcount_scale,
                 # 17 classes + background; the base still had the 16-class values.
                 semantic_bev_cache_shape=(18, 50, 50),
                 semantic_bev_cache_projection=dict(num_classes=17))

semantic_bev_cache_settings = dict(**_semantic)

data = dict(train=dict(**_semantic),
            val=dict(**_semantic),
            test=dict(**_semantic))
