# Stage 1: frozen DINOv3 BEV targets, semantic spatial conditions and concatenated CLS global conditions.
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

_dim_ = 384
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1

bev_h_ = 50
bev_w_ = 50
queue_length = 3
use_temporal_condition = True

num_bboxes = 300
num_classes = len(class_names) + 2
use_3d_bbox = True
use_semantics = True

use_depth = True  # Metric3D raw metric depth for semantic surface splatting
metric_depth_range = (2.5, 170.0)
use_semantic_bev_cache = True
semantic_bev_cache_root = '../../data/semantic_bev'

# Shared camera order for dataset inputs and CLS concatenation.
camera_types = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# Semantic spatial conditions and camera-concatenated CLS global embedding.
unet = dict(
    type='layout_diffusion.seg_diffusion_unet_v5_dino.DiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_global_cls=True, # CLS embedding
        cls_channels=_dim_,
        use_temporal_condition=use_temporal_condition,
        temporal_history_length=queue_length - 1,
        temporal_feature_channels=384, # DINOv3
        cls_camera_order=camera_types,
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
        attention_ds=[4, 2, 1],
        channel_mult=[1, 2, 4],
        dropout=0.0,
        use_checkpoint=False,
        seg_bev_aligner=dict(
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            sky_as_ignore=True,
            num_classes=16,
            emb_channels=256,
            channel_mult=[1, 1, 1],
            depth_range=metric_depth_range,
            depth_type='camera_z',
            splat_mode='bilinear',
            pixel_stride=1,
        ),
    ),
)

# Only the UNet is optimized. No BEVFormer checkpoint or detection head is used.
# Photometric distortion is applied consistently with the current DINO experiment.
# The loader normalizes/resizes RGB and aligns depth to each temporal frame.
dino_bev_extractor = dict(
    encoder=dict(
        model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        revision='main',  # Resolved HF commit is recorded and checked on resume.
        cache_dir=None,
        local_files_only=False,
        view_batch_size=6,
        inference_dtype='float32',
    ),
    projector=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        depth_range=metric_depth_range,
        depth_type='camera_z',
        splat_mode='bilinear',
        pixel_stride=1,
        point_chunk_size=8192,
    ),
)

dataset_type = 'CustomNuScenesDiffusionDataset_seg_depth_v2'
data_root = '../../data/nuscenes/'
semantic_root = '../../data/nuscenes_sam3'
depth_root = '../../data/nuscenes_metric3d'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img'])
]


val_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
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
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

semantic_bev_cache_settings = dict(
    use_semantic_bev_cache=use_semantic_bev_cache,
    load_depth_in_cache=True,
    load_temporal_depth=use_temporal_condition,
    filter_empty_gt=False,
    semantic_bev_cache_root=semantic_bev_cache_root,
    semantic_bev_cache_shape=(unet['parameters']['seg_bev_aligner']['num_classes'] + 1,
                              bev_h_, bev_w_),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        camera_types=camera_types,
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
             camera_types=camera_types,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             use_semantics=use_semantics,
             use_depth=use_depth,
             semantic_path=semantic_root,
             seg_id_remap=seg_id_remap,  # SAM3 raw id -> model taxonomy
             depth_path=depth_root,
             depth_raw_shape=(900, 1600),
             depth_range=metric_depth_range,
             test_mode=False, queue_length=queue_length,
             pipeline=val_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             **semantic_bev_cache_settings),
    test=dict(type=dataset_type,
              camera_types=camera_types,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              use_semantics=use_semantics,
              use_depth=use_depth,
              semantic_path=semantic_root,
              seg_id_remap=seg_id_remap,  # SAM3 raw id -> model taxonomy
              depth_path=depth_root,
              depth_raw_shape=(900, 1600),
              depth_range=metric_depth_range,
              test_mode=False, queue_length=queue_length,
              pipeline=val_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, **semantic_bev_cache_settings),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
