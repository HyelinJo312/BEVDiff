_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin.datasets.occ3d_nuscenes_dataset',
        'projects.mmdet3d_plugin.bevformer.dense_heads.occ3d_head',
        'projects.bevdiffuser.occ_task',
    ],
    allow_failed_imports=False)

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]
occ_voxel_size = [0.4, 0.4, 0.4]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)

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
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4
num_occ_classes = 18
num_classes = num_occ_classes
num_bboxes = 0
use_3d_bbox = False
use_semantics = True
use_depth = True  # DA3 depth maps for FB-BEV depth consistency in BEV aligners

# Seg-only UNet
unet = dict(
    type='layout_diffusion.seg_diffusion_unet_v2.DiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        return_multiscale=False,
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
        # ---- Segmentation one-hot → BEV aligner (with FB-BEV depth consistency) ----
        seg_bev_aligner=dict(
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            sky_as_ignore=True,
            num_points_in_pillar=4,
            pillar_z_range=(-1.0, 1.16),
            num_classes=16,
            seg_downsample_factor=2,  # 1=full-res semantic one-hot, 2=current half-res setting
            emb_channels=256,
            channel_mult=[1, 1, 1],  # keep 256ch at every scale; must match seg_channels above
            final_dim=(480, 800),  # H x W after RandomScaleImageMultiViewImage(0.5) + PadMultiViewImage(32)
            depth_consistency_mode='gaussian',  # gaussian | bin_linear | None
            depth_consistency_sigma=4.0,
        ),
    ),
)

model = dict(
    type='BEVFormerOccDiffusion',
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
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerOccEncoder',
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        embed_dims=_dim_,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
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
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm'))),
            # PerceptionTransformer builds a decoder at init time, but this
            # occupancy-only encoder never calls it.
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=1,
                return_intermediate=False,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_)),
    # Task path matches configs/occ3d/bevformer_base_occ.py after the BEV.
    # The backbone, image pipeline and 6-layer BEV encoder above are unchanged.
    occ_head=dict(
        type='DiffusionOccHead',
        in_channels=_dim_,
        pillar_h=occ_size[2],
        out_dim=32,
        num_classes=num_occ_classes,
        use_mask=False,
        loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

dataset_type = 'Occ3DDiffusionNuScenesDataset'
data_root = 'data/nuscenes/'
occ_root = 'data/occ3d/gts'
semantic_root = 'data/nuscenes_sam3'
depth_root = 'data/nuscenes_depth_da3'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccGTForDiffusion', occ_root=occ_root),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names,
         with_label=False),
    dict(type='CustomCollect3D',
         keys=['voxel_semantics', 'mask_lidar', 'mask_camera', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=class_names,
                 with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_root,
        pc_range=point_cloud_range,
        num_classes=num_occ_classes,
        use_semantics=use_semantics,
        use_depth=use_depth,
        semantic_path=semantic_root,
        depth_path=depth_root,
        seg_id_remap=seg_id_remap,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        filter_empty_gt=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_root,
        pc_range=point_cloud_range,
        num_classes=num_occ_classes,
        use_semantics=use_semantics,
        use_depth=use_depth,
        semantic_path=semantic_root,
        depth_path=depth_root,
        seg_id_remap=seg_id_remap,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_root,
        pc_range=point_cloud_range,
        num_classes=num_occ_classes,
        use_semantics=use_semantics,
        use_depth=use_depth,
        semantic_path=semantic_root,
        depth_path=depth_root,
        seg_id_remap=seg_id_remap,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=12, pipeline=test_pipeline)

load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
