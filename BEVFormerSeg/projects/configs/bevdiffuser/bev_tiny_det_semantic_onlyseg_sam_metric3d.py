# BEVFormer-tiny + BEVDiffuser, semantic-BEV-prior-only condition, SEG task.
#
#   Diffusion condition : semantic BEV prior ONLY (no layout encoder, no DINO)
#   Downstream task head: BEV segmentation (BEVFusionSegmentationV2, 6 cls + bg)


_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# BEV feature grid vs. seg label canvas. BevFeatureSlicer resamples the 50x50
# BEV onto the map canvas ((200, 400) at 0.15 m).
det_grid_conf = {
    'xbound': [-51.2, 51.2, 0.68],
    'ybound': [-51.2, 51.2, 0.68],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}

map_grid_conf = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}

grid_conf = map_grid_conf


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Segmentation TASK classes (BEVFusion-style filled polygons).
#   0: background, 1: drivable_area, 2: carpark_area, 3: walkway,
#   4: stop_line, 5: ped_crossing, 6: divider
seg_classes = ('drivable_area', 'ped_crossing', 'walkway',
               'stop_line', 'carpark_area', 'divider')

# SAM3 raw seg-id -> model taxonomy remap
#   lane(17)/road arrow(18) -> road(2)
#   sky(19) -> 17
seg_id_remap = {17: 2, 18: 2, 19: 17}

num_seg_classes = len(seg_classes) + 1   # +1 for background


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3 # each sequence contains `queue_length` frames.

num_bboxes = 300
num_classes = len(class_names) + 2
use_3d_bbox = True
use_semantics = False  # cached semantic BEV replaces raw SAM3 maps

use_depth = False  # Metric3D depth is already included in the shared BEV cache
use_semantic_bev_cache = True
metric_depth_range = (2.5, 170.0)
# Counts cache (layout share+counts+support = [18 share | 18 counts | 1 support]).
# semantic_type='share' uses only the share (class distribution) channels [:18];
# set 'share+logcount' to also feed log1p(counts) (see *_logcount.py).
semantic_bev_cache_root = '../../data/semantic_bev_counts'
semantic_type = 'share'
semantic_logcount_scale = 1e5

# Seg-only UNet: no DINO / layout condition.
unet = dict(
    type='projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5_4.DiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        sbam_use_scale_shift_norm=False,
        use_global_seg=True,  # GAP+MLP global semantics added to time embedding and gates
        in_channels=_dim_,
        out_channels=_dim_,
        seg_channels=[256, 256, 256],
        model_channels=256,
        num_head_channels=32,
        num_heads=-1,
        num_heads_upsample=-1,
        num_res_blocks=2,
        num_attention_blocks=1,
        resblock_updown=True,
        use_spatial_transformer=True,
        num_pre_downsample=0,
        attention_ds=[ 4, 2, 1 ],
        channel_mult=[ 1, 2, 4 ],
        dropout=0.0,
        use_checkpoint=False,
        # ---- Metric3D semantic surface -> BEV push-style splatting ----
        seg_bev_aligner=dict(
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            sky_as_ignore=True,
            num_classes=17,
            emb_channels=256,
            channel_mult=[1, 1, 1],  # keep 256ch at every scale; must match seg_channels above
            depth_range=metric_depth_range,
            depth_type='camera_z',  # Metric3D optical-axis depth; ray_distance is also supported
            splat_mode='bilinear',
            pixel_stride=1,
            semantic_type=semantic_type,
            logcount_scale=semantic_logcount_scale,
        ),
    ),
)


model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    # pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=False,
        as_two_stage=False,
        task={'seg': True,
              'det': False},
        det_grid_conf=det_grid_conf,
        map_grid_conf=map_grid_conf,
        # Head architecture identical to bevformer_tiny_seg_v3.py so that
        # results/bevformer_tiny_seg_v3/epoch_24.pth loads without shape errors.
        seg_encoder=dict(
            type='SegEncode',
            inC=256,
            outC=num_seg_classes),
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=3.0,
            class_weight=[0.2,   # 0 background
                          0.5,   # 1 drivable_area
                          2.0,   # 2 carpark_area
                          2.0,   # 3 walkway
                          2.0,   # 4 stop_line
                          2.0,   # 5 ped_crossing
                          2.0]), # 6 divider
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        # seg 전용이라 bbox 를 뽑지는 않지만, BEVFormerHead 가 bbox_coder 에서
        # pc_range 를 읽으므로 정의는 남겨둔다.
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))


dataset_type = 'CustomNuScenesDiffusionDataset_seg_depth_v2'
data_root = '../../data/nuscenes/'
semantic_root = '../../data/nuscenes_sam3'
depth_root = '../../data/nuscenes_metric3d'
file_client_args = dict(backend='disk')


semantic_bev_cache_settings = dict(
    use_semantic_bev_cache=use_semantic_bev_cache,
    semantic_bev_cache_root=semantic_bev_cache_root,
    seg_id_remap=seg_id_remap,
    semantic_bev_cache_shape=(unet['parameters']['seg_bev_aligner']['num_classes'] + 1,
                              bev_h_, bev_w_),
    # Checked against the cache manifest at startup (grid and taxonomy must match).
    semantic_bev_cache_projection=dict(
        bev_h=bev_h_, bev_w=bev_w_, pc_range=point_cloud_range,
        num_classes=unet['parameters']['seg_bev_aligner']['num_classes'],
        sky_as_ignore=True),
    semantic_type=semantic_type,
    semantic_logcount_scale=semantic_logcount_scale,
)


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # BEV segmentation GT -> results['semantic_indices']  (200 x 400)
    dict(type='BEVFusionSegmentationV2', dataset_root=data_root, map_grid_conf=map_grid_conf, classes=seg_classes),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'semantic_indices'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='BEVFusionSegmentationV2',
         dataset_root=data_root,
         map_grid_conf=map_grid_conf,
         classes=seg_classes),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'semantic_indices'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        use_semantics=use_semantics,
        use_depth=use_depth,
        # semantic_path=data_root + 'nuscenes_semantic_sam3',
        semantic_path=None,
        **semantic_bev_cache_settings,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        grid_conf=grid_conf,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             use_semantics=use_semantics,
             use_depth=use_depth,
            #  semantic_path=data_root + 'nuscenes_semantic_sam3',
             semantic_path=None,
             **semantic_bev_cache_settings,
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             grid_conf=grid_conf,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              use_semantics=use_semantics,
              use_depth=use_depth,
            #   semantic_path=data_root + 'nuscenes_semantic_sam3',
              semantic_path=None,
              **semantic_bev_cache_settings,
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              grid_conf=grid_conf,
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=12, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=2)
