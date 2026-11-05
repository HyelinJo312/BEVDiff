# Detection semantic BEV with counts, keeping crosswalk as its own class.
# SAM3 raw seg-id -> model taxonomy:
#   crosswalk(16) kept as class 16; lane(17)/road arrow(18) -> road(2); sky(19) -> 17
# num_classes=17 so that 17 (sky) is the ignored id; with num_classes=16 the
# crosswalk id would equal the sky id and be dropped.
# Stored cache: [18 share | 18 counts | 1 support] = 37 channels; model input
# for semantic_type='share+logcount' is 36 channels.
# Build config for BEVFormerSeg/data/semantic_bev_counts (BEV map segmentation).
_base_ = ['./bev_tiny_onlyseg_sam_v2_metric3d_logcount.py']

seg_id_remap = {'_delete_': True, 17: 2, 18: 2, 19: 17}
semantic_bev_cache_root = '../../data/semantic_bev_counts'

unet = dict(parameters=dict(seg_bev_aligner=dict(num_classes=17)))

_semantic = dict(seg_id_remap=seg_id_remap,
                 semantic_bev_cache_root=semantic_bev_cache_root,
                 semantic_bev_cache_shape=(18, 50, 50))
data = dict(train=dict(**_semantic), val=dict(**_semantic), test=dict(**_semantic))
