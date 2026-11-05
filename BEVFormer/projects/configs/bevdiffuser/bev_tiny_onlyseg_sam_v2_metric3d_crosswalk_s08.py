# Build config for BEVFormerSeg/data/semantic_bev_counts (BEV map segmentation).
# Based on bev_tiny_onlyseg_sam_v2_metric3d.py with only what the BEVFormerSeg config
# bev_tiny_det_semantic_onlyseg_sam_metric3d.py requires from the cache:
#   - taxonomy: crosswalk(16) kept as class 16; lane(17)/road arrow(18) -> road(2);
#     sky(19) -> 17 = ignored id (num_classes=17, 18 share channels)
#   - stored layout share+counts+support (37 channels); its share channels are the
#     plain share input, and the counts keep the seg logcount variant usable
#   - image scale 0.8 (the BEVFormerSeg RandomScaleImageMultiViewImage) instead of 0.5;
#     the builder takes each split's image geometry from these pipelines and drops
#     photometric and annotation steps.
_base_ = ['./bev_tiny_onlyseg_sam_v2_metric3d.py']

image_scale = 0.8

seg_id_remap = {'_delete_': True, 17: 2, 18: 2, 19: 17}
semantic_bev_cache_layout = 'share+counts+support'
semantic_bev_cache_root = '../../../BEVFormerSeg/data/semantic_bev_counts'

unet = dict(parameters=dict(seg_bev_aligner=dict(num_classes=17)))

# Same values as the base config; pipelines are replaced, not merged.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[image_scale]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[image_scale]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
        ])
]

_semantic = dict(seg_id_remap=seg_id_remap,
                 semantic_bev_cache_root=semantic_bev_cache_root,
                 semantic_bev_cache_shape=(18, 50, 50))
data = dict(train=dict(pipeline=train_pipeline, **_semantic),
            val=dict(pipeline=test_pipeline, **_semantic),
            test=dict(pipeline=test_pipeline, **_semantic))
evaluation = dict(pipeline=test_pipeline)
