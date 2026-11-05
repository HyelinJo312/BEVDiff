# Stage 1 (semantic-only BEV diffusion teacher) on frozen BEVDet4D-R50 (C0) BEV features.
#
# Frozen target: configs/bevdet/bevdet-r50-4d-cbgs.py model (unchanged; bevdet-r50-4d.py has the
#   same model), weights = our C0 EMA passed by the launch script, chosen by its RECIPE:
#   nocbgs -> bevdet4d-r50-nocbgs_seed0/epoch_24_ema.pth, cbgs -> bevdet4d-r50-cbgs_seed0/epoch_20_ema.pth.
#   The base config is used for the model and val/test only; data.train below never uses CBGS.
#   x0 = BEVDet4D.extract_img_feat(img_inputs, img_metas)[0][0]  -> [B, 256, 128, 128]
# UNet: identical to the flagship Stage 1 (bev_tiny_onlyseg_sam_v2_metric3d.py, gated SBAM v5)
#   except the resolution keys: image_size=128, num_pre_downsample=1 (UNet core 64/32/16),
#   seg_bev_aligner bev_h=bev_w=64 and ds4_padding=1 (prior scales 64/32/16).
# Data (decisions 2026-11-04): no CBGS; BEVDet4D train augmentation unchanged (image aug +
#   BEV aug). The 50x50 LiDAR-frame semantic cache is warped into the augmented key-ego
#   frame and resampled to 64x64 by LoadSemanticBEVPrior (inserted right after BEVAug).
# Paths are relative to BEVDiff/BEVDet (the launch script runs from there).

_base_ = ['../../../configs/bevdet/bevdet-r50-4d-cbgs.py']

custom_imports = dict(imports=['projects.bevdiffuser.bevdet_semantic_prior'],
                      allow_failed_imports=False)

# ---- copied from the base config (the train pipeline below must match it) ----
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
multi_adj_frame_id_cfg = (1, 1 + 1, 1)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# ---- Stage 1 additions ----
_dim_ = 256
bev_h_ = 128   # BEVDet grid: x, y in [-51.2, 51.2] at 0.8 m
bev_w_ = 128
prior_h_ = 64  # UNet core / prior grid (one pre-downsample)
prior_w_ = 64
semantic_bev_cache_root = '../BEVFormer/data/semantic_bev'
semantic_bev_cache_shape = (17, 50, 50)   # num_classes(16) + 1, LiDAR frame (manifest-checked)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadSemanticBEVPrior',
        cache_root=semantic_bev_cache_root,
        cache_shape=semantic_bev_cache_shape,
        out_size=(prior_h_, prior_w_),
        semantic_type='share'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'semantic_bev_probabilities'])
]

data = dict(
    train=dict(
        _delete_=True,   # drop the base CBGSDataset wrapper: Stage 1 uses the natural distribution
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg))

unet = dict(
    type='layout_diffusion.seg_diffusion_unet_v5.DiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_global_seg=True,
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
        num_pre_downsample=1,
        attention_ds=[4, 2, 1],
        channel_mult=[1, 2, 4],
        dropout=0.0,
        use_checkpoint=False,
        seg_bev_aligner=dict(
            bev_h=prior_h_,
            bev_w=prior_w_,
            pc_range=point_cloud_range,
            sky_as_ignore=True,
            num_classes=16,
            emb_channels=256,
            channel_mult=[1, 1, 1],
            depth_range=(2.5, 170.0),
            depth_type='camera_z',
            splat_mode='bilinear',
            pixel_stride=1,
            use_legacy_prob_embed=False,
            ds4_padding=1,
        ),
    ),
)
