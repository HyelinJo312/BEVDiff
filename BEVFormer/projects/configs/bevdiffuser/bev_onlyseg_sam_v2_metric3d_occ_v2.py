# Despite the legacy filename, this uses the full R101 / 200x200 occupancy baseline.
# Run entrypoints from BEVFormer/. The UNet uses a 50x50 semantic condition.
# Standalone config: all occupancy model/data/pipeline settings below are inlined
# (previously inherited from ../occ3d/bevformer_base_occ.py) so this file is
# self-contained, mirroring bev_tiny_onlyseg_sam_v2_metric3d.py.
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# SAM3 raw seg-id -> model taxonomy remap
#   crosswalk(16)/lane(17)/road arrow(18) -> road(2)
#   sky(19) -> 16
seg_id_remap = {16: 2, 17: 2, 18: 2, 19: 16}

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4  # each sequence contains `queue_length` frames.

num_bboxes = 0
num_classes = 18
use_3d_bbox = False

metric_depth_range = (2.5, 170.0)

use_semantic_bev_cache = True
semantic_bev_cache_root = '../../data/semantic_bev_occ_v2'
semantic_bev_cache_source_version = 'semantic_bev_occ_ego'

# Seg-only UNet with full-resolution local refinement.
unet = dict(
    type='projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5_4.DiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_global_seg=True,  # GAP+MLP global semantics added to time embedding and gates
        use_local_refinement=True,
        local_refinement_channels=64,
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
        num_pre_downsample=2,
        attention_ds=[ 4, 2, 1 ],
        channel_mult=[ 1, 2, 4 ],
        dropout=0.0,
        use_checkpoint=False,
        # ---- Metric3D semantic surface -> BEV push-style splatting ----
        seg_bev_aligner=dict(
            bev_h=50,
            bev_w=50,
            pc_range=point_cloud_range,
            sky_as_ignore=True,
            num_classes=16,
            emb_channels=256,
            channel_mult=[1, 1, 1],  # keep 256ch at every scale; must match seg_channels above
            depth_range=metric_depth_range,
            depth_type='camera_z',  # Metric3D optical-axis depth; ray_distance is also supported
            splat_mode='bilinear',
            pixel_stride=1,
            output_frame='ego',
        ),
    ),
)

# This subclass adds only BEV-only extraction/decoding entrypoints. All modules,
# parameter names, normalization, GT IDs and loss settings are inherited unchanged
# from the R101/DCN, 200x200 Occ3D baseline (projects/configs/occ3d/bevformer_base_occ.py).
model = dict(
    type='BEVFormerOcc',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerOccHead',
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=18,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_mask=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        transformer=dict(
            type='TransformerOcc',
            pillar_h=16,
            num_classes=18,
            norm_cfg=dict(type='BN', ),
            norm_cfg_3d=dict(type='BN3d', ),
            use_3d=True,
            use_conv=False,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoderOcc',
                num_layers=4,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayerOcc',
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
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
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
                pc_range=point_cloud_range))),
    ),
)

dataset_type = 'Occ3DDiffusionNuScenesDataset'
data_root = '../../data/occ3d-nus/'
semantic_path = '../../data/nuscenes_sam3'
depth_path = '../../data/nuscenes_metric3d'
file_client_args = dict(backend='disk')
occ_gt_data_root = '../../data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

occ_conditions = dict(
    type=dataset_type,
    pc_range=point_cloud_range,
    use_semantics=True, use_depth=True,
    semantic_path=semantic_path,
    depth_path=depth_path,
    seg_id_remap=seg_id_remap,
    depth_raw_shape=(900, 1600),
    depth_range=metric_depth_range,
    use_semantic_bev_cache=use_semantic_bev_cache,
    semantic_bev_cache_root=semantic_bev_cache_root,
    semantic_bev_cache_output_frame='ego',
    semantic_bev_cache_shape=(17, 50, 50)
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        **occ_conditions),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        box_type_3d='LiDAR',
        test_mode=True,
        samples_per_gpu=1,
        **occ_conditions),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        bev_size=(bev_h_, bev_w_),
        box_type_3d='LiDAR',
        test_mode=True,
        **occ_conditions),
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
    warmup_iters=500,  # Original setting for samples_per_gpu=1.
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=6, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=2)
