# GT-layout BEVDiffuser Stage 1 for the native BEVFormer Occ3D baseline.
_base_ = ['../occ3d/bevformer_base_occ.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]
num_occ_classes = 18

_dim_ = 256
bev_h_ = 200
bev_w_ = 200
queue_length = 4

# The layout taxonomy remains the original ten nuScenes object classes plus
# the whole-image and null sentinel classes.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_bboxes = 300
num_classes = len(class_names) + 2
use_3d_bbox = True
use_layout = True

unet = dict(
    type='projects.bevdiffuser.layout_diffusion.layout_diffusion_unet.LayoutDiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        in_channels=_dim_,
        out_channels=_dim_,
        model_channels=256,
        encoder_channels=256,
        num_head_channels=32,
        num_heads=-1,
        num_heads_upsample=-1,
        num_res_blocks=2,
        num_attention_blocks=1,
        resblock_updown=True,
        # 200 -> 100 -> 50 before the original three-level UNet.
        num_pre_downsample=2,
        attention_ds=[4, 2, 1],
        channel_mult=[1, 2, 4],
        dropout=0.0,
        use_checkpoint=False,
        use_positional_embedding_for_attention=True,
        attention_block_type='ObjectAwareCrossAttention',
        layout_encoder=dict(
            type='projects.bevdiffuser.layout_diffusion.layout_encoder.LayoutTransformerEncoder',
            parameters=dict(
                used_condition_types=['obj_class', 'obj_bbox', 'is_valid_obj'],
                layout_length=num_bboxes,
                num_classes_for_layout_object=num_classes,
                mask_size_for_layout_object=0,
                hidden_dim=256,
                output_dim=1024,
                num_layers=6,
                num_heads=8,
                use_final_ln=True,
                use_positional_embedding=False,
                resolution_to_attention=[12, 25, 50],
                use_key_padding_mask=False,
                use_3d_bbox=use_3d_bbox))))

# Use the native detector; diffusion entrypoints are implemented directly on it.
model = dict(type='BEVFormerOcc')

stage1_bev_checkpoint = '../results/occ3d/bevformer_base_occ/epoch_24.pth'

dataset_type = 'Occ3DLayoutDiffusionNuScenesDataset'
data_root = 'data/occ3d-nus/'
occ_gt_data_root = 'data/occ3d-nus/'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D',
         keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
               'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names,
                 with_label=False),
            dict(type='CustomCollect3D',
                 keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                       'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

layout_dataset_args = dict(
    type=dataset_type,
    data_root=data_root,
    pc_range=point_cloud_range,
    use_3d_bbox=use_3d_bbox,
    num_classes=num_classes,
    num_bboxes=num_bboxes,
    use_layout=use_layout,
    classes=class_names,
    modality=input_modality,
    use_valid_flag=True,
    bev_size=(bev_h_, bev_w_),
    queue_length=queue_length,
    box_type_3d='LiDAR')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        **layout_dataset_args),
    val=dict(
        _delete_=True,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        load_annos=True,
        samples_per_gpu=1,
        **layout_dataset_args),
    test=dict(
        _delete_=True,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        load_annos=True,
        **layout_dataset_args),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

evaluation = dict(interval=1, pipeline=test_pipeline)
