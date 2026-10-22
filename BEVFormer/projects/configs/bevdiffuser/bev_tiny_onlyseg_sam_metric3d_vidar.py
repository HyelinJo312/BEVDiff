# Stage 1 ViDAR target variant. ViDAR is built in an isolated worker because
# its plugin registry collides with BEVFormer's. The worker builds the model
# defined here and uses ViDAR's native test image preprocessing.
# multi-scale feautres -> single scale features (C5)


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

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50

vidar_bev_h = 200
vidar_bev_w = 200
queue_length = 3 # each sequence contains `queue_length` frames.

num_bboxes = 300
num_classes = len(class_names) + 2
use_3d_bbox = True
use_semantics = True

use_depth = True  # Metric3D raw metric depth for semantic surface splatting
metric_depth_range = (2.5, 170.0)
use_semantic_bev_cache = True
semantic_bev_cache_root = '../../data/semantic_bev'

# Seg-only UNet: no DINO / layout condition.
unet = dict(
    type='layout_diffusion.seg_diffusion_unet_v5.DiffusionUNetModel',
    parameters=dict(
        image_size=vidar_bev_h,
        use_fp16=False,
        use_scale_shift_norm=True,
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
        num_pre_downsample=2,
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
            num_classes=16,
            emb_channels=256,
            channel_mult=[1, 1, 1],  # keep 256ch at every scale; must match seg_channels above
            depth_range=metric_depth_range,
            depth_type='camera_z',  # Metric3D optical-axis depth; ray_distance is also supported
            splat_mode='bilinear',
            pixel_stride=1,
        ),
    ),
)


# Frozen ViDAR model. Keep its BEV grid and FPN scales separate from the
# 50x50 semantic condition above. This definition matches the ViDAR checkpoint.
vidar_num_levels = 4
vidar_queue_length = 4
future_pred_frame_num_train = 0
future_pred_frame_num_test = 0
future_decoder_layer_num = 1
frame_loss_weight = [[1], [0]]
vidar_head_pred_history_frame_num = 3
vidar_head_pred_future_frame_num = 1
vidar_head_per_frame_loss_weight = (0.2, 0.4, 0.6, 1.0, 1.2)
latent_render_act_fn = 'sigmoid'
latent_render_layer_idx = (2,)
latent_render_grid_step = 0.5

model = dict(
    type='ViDAR',
    use_grid_mask=True,
    video_test_mode=True,
    # BEV configuration.
    point_cloud_range=point_cloud_range,
    bev_h=vidar_bev_h,
    bev_w=vidar_bev_w,
    # Predict frame num.
    future_pred_frame_num=future_pred_frame_num_train,
    test_future_frame_num=future_pred_frame_num_test,
    random_drop_prev_rate=0.1,
    random_drop_prev_end_idx=3,
    backwarded_prev_frame_num=1,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3,),
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
    future_pred_head=dict(
        type='ViDARHeadV1',
        history_queue_length=vidar_queue_length,
        pred_history_frame_num=vidar_head_pred_history_frame_num,
        pred_future_frame_num=vidar_head_pred_future_frame_num,
        per_frame_loss_weight=vidar_head_per_frame_loss_weight,
        ray_grid_num=512,
        ray_grid_step=1.0,
        use_ce_loss=True,
        use_dist_loss=False,
        use_dense_loss=True,
        num_pred_fcs=0,  # head for point cloud prediction.
        num_pred_height=16,  # Predict BEV instead of 3D space occupancy.
        can_bus_norm=True,
        can_bus_dims=(0, 1, 2, 17),
        bev_h=vidar_bev_h,
        bev_w=vidar_bev_w,
        pc_range=point_cloud_range,
        loss_weight=frame_loss_weight,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=vidar_bev_h,
            col_num_embed=vidar_bev_w,
        ),
        transformer=dict(
            type='PredictionTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='PredictionDecoder',
                num_layers=future_decoder_layer_num,
                return_intermediate=True,
                transformerlayers=dict(
                    type='PredictionTransformerLayer',
                    attn_cfgs=[
                        # layer-1: deformable self-attention.
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                        # layer-2: deformable cross-attention,
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )),
    ),
    pts_bbox_head=dict(
        type='ViDARBEVFormerHead',
        bev_h=vidar_bev_h,
        bev_w=vidar_bev_w,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='CustomBEVFormerEncoder',
                keep_idx=latent_render_layer_idx,
                latent_rendering_lid=latent_render_layer_idx,
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayerV2',
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
                                num_levels=vidar_num_levels),
                            embed_dims=_dim_,
                        )
                    ],
                    latent_render=dict(embed_dims=256, pred_height=16, num_pred_fcs=0,
                                       grid_step=latent_render_grid_step, grid_num=256,
                                       reduction=16, act=latent_render_act_fn,),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'latent_render', 'ffn', 'norm'))),
            # !!!!!!! DECODER NOT USED !!!!!!!
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
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
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
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
            row_num_embed=vidar_bev_h,
            col_num_embed=vidar_bev_w,
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
train_dataset_type = 'ViDARSemanticCacheDataset'
data_root = '../../data/nuscenes/'
semantic_root = '../../data/nuscenes_sam3'
depth_root = '../../data/nuscenes_metric3d'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_labels_3d'], meta_keys=('sample_idx',))
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args,),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'points'])
]

semantic_bev_cache_settings = dict(
    use_semantic_bev_cache=use_semantic_bev_cache,
    semantic_bev_cache_root=semantic_bev_cache_root,
    semantic_bev_cache_shape=(unet['parameters']['seg_bev_aligner']['num_classes'] + 1,
                              bev_h_, bev_w_),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=train_dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        use_semantics=use_semantics,
        use_depth=use_depth,
        semantic_path=semantic_root,
        seg_id_remap=seg_id_remap,  # SAM3 raw id -> model taxonomy
        depth_path=depth_root,
        depth_raw_shape=(900, 1600),
        depth_range=metric_depth_range,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR', **semantic_bev_cache_settings),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             use_semantics=use_semantics,
             use_depth=use_depth,
             semantic_path=semantic_root,
             seg_id_remap=seg_id_remap,  # SAM3 raw id -> model taxonomy
             depth_path=depth_root,
             depth_raw_shape=(900, 1600),
             depth_range=metric_depth_range,
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             **semantic_bev_cache_settings),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              use_semantics=use_semantics,
              use_depth=use_depth,
              semantic_path=semantic_root,
              seg_id_remap=seg_id_remap,  # SAM3 raw id -> model taxonomy
              depth_path=depth_root,
              depth_raw_shape=(900, 1600),
              depth_range=metric_depth_range,
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, **semantic_bev_cache_settings),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
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
