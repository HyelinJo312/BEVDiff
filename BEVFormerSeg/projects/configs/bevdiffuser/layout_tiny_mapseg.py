# Stage 1 — BEVDiffuser (GT layout condition) retrained on the segmentation BEV.
# Baseline for BEV map segmentation, adapted with the minimum changes:
#   - kept from BEVDiffuser: GT 3D-box layout condition and the layout UNet
#     (identical to BEVFormer/projects/configs/bevdiffuser/layout_tiny.py),
#     task loss through the task head (--task_loss_scale 0.1).
#   - changed for the task: x0 and the task head come from the segmentation
#     BEVFormer-tiny (results/map_seg/baseline/bevformer_tiny_seg_v3/epoch_24.pth);
#     model, image pipeline and seg classes below match that checkpoint.
# Optimizer, batch size and steps are set by projects/bevdiffuser/train_original.sh.

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

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

# BEVFusion seg classes for the student's seg head
# (matches bevformer_tiny_seg_v2.py / BEVFusionSegmentationV2)
seg_classes = ('drivable_area', 'ped_crossing', 'walkway',
               'stop_line', 'carpark_area', 'divider')

num_seg_classes = len(seg_classes) + 1  # +1 for background


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
queue_length = 3  # each sequence contains `queue_length` frames.

num_bboxes = 300
num_classes = len(class_names) + 2
use_3d_bbox = True
use_layout = True
use_semantics = False


unet = dict(
    type='projects.bevdiffuser.layout_diffusion.layout_diffusion_unet.LayoutDiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        in_channels=_dim_,
        out_channels=_dim_,
        model_channels=256,
        encoder_channels=256, # assert same as layout_encoder.hidden_dim
        num_head_channels=32,
        num_heads=-1,
        num_heads_upsample=-1,
        num_res_blocks=2,
        num_attention_blocks=1,
        resblock_updown=True,
        attention_ds=[ 4, 2, 1 ],
        channel_mult=[ 1, 2, 4 ],
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
                output_dim=1024, # model_channels x 4
                num_layers=6,
                num_heads=8,
                use_final_ln=True,
                use_positional_embedding=False,
                resolution_to_attention=[12, 25, 50], #[ 8, 16, 32 ],
                use_key_padding_mask=False,
                use_3d_bbox=use_3d_bbox),
            ),
    ),
)

# ── Frozen BEV model (x0 source and seg task head) ─────────────────────────
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
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
        task={'seg': True, 'det': False},
        det_grid_conf=det_grid_conf,
        map_grid_conf=map_grid_conf,
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
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset_type = 'CustomNuScenesDiffusionDataset_layout'
data_root = '../../data/nuscenes/'  # Stage 1 runs from projects/bevdiffuser
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
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
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        use_layout=use_layout,
        use_semantics=use_semantics,
        # semantic_path=data_root + 'nuscenes_semantic_v2',
        semantic_path=data_root + 'nuscenes_sam3',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        grid_conf=grid_conf,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             use_layout=use_layout,
             use_semantics=use_semantics,
            #  semantic_path=data_root + 'nuscenes_semantic_v2',
             semantic_path=data_root + 'nuscenes_sam3',
             pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
             grid_conf=grid_conf,
             classes=class_names,
             modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              use_layout=use_layout,
              use_semantics=use_semantics,
            #   semantic_path=data_root + 'nuscenes_semantic_v2',
              semantic_path=data_root + 'nuscenes_sam3',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              grid_conf=grid_conf,
              classes=class_names,
              modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
