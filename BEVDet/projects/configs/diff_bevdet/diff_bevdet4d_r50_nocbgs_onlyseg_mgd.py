# Stage 2 (no-CBGS recipe): BEVDet4D-R50 student distilled from the semantic-only BEV diffusion teacher.
#
# Identical to diff_bevdet4d_r50_cbgs_onlyseg_mgd.py (flagship teacher / MGD settings: student from
# scratch, load_from = frozen target only, t=100, DDIM 5, CFG 2, MGD lambda 0.55 alpha 100,
# loss 0.5*task + 0.5*bev) except for the C0 recipe it follows (added 2026-11-05):
#   - _base_ = configs/bevdet/bevdet-r50-4d.py: plain NuScenesDataset (no CBGS), 24 epochs, lr step [24]
#   - frozen target = the no-CBGS C0 EMA (bevdet4d-r50-nocbgs_seed0/epoch_24_ema.pth)
#   - teacher = the Stage 1 UNet trained on that target (Stage 1 RECIPE=nocbgs)
# Launch: RECIPE=nocbgs bash tools/dist_train_diff_bevdet4d.sh (4 GPUs x 8 = batch 32, as the C0 run).
# Evaluate with tools/dist_test_bevdet4d.sh (CONFIG=this file; always --no-aavt).
# Paths are relative to BEVDiff/BEVDet.

_base_ = ['../../../configs/bevdet/bevdet-r50-4d.py']

custom_imports = dict(
    imports=['projects.bevdet_plugin', 'projects.bevdiffuser.bevdet_semantic_prior'],
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

# ---- semantic prior (same as Stage 1: bevdiffuser/bevdet4d_r50_onlyseg_sam_v2_metric3d.py) ----
_dim_ = 256
bev_h_ = 128
bev_w_ = 128
prior_h_ = 64
prior_w_ = 64
semantic_bev_cache_root = '../BEVFormer/data/semantic_bev'
semantic_bev_cache_shape = (17, 50, 50)

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

# plain NuScenesDataset of the base (no CBGS, C0 recipe); only its pipeline gains the prior step
data = dict(train=dict(pipeline=train_pipeline))

model = dict(
    type='DiffBEVDet4D',
    use_mgd=True,
    mgd_alpha=100,
    mgd_lambda=0.55,
    loss_mix_weight=0.5)

# Stage 1 UNet (gated SBAM v5): identical to the Stage 1 config except the absolute import path
# (build_unet_v2 imports `type` as a module path; Stage 1 ran from projects/bevdiffuser).
unet = dict(
    type='projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5.DiffusionUNetModel',
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
            use_legacy_prob_embed=True,
            ds4_padding=1,
        ),
    ),
)

bev_diffuser_cfg = dict(
    unet_cfg=unet,  # replaced by the top-level `unet` in tools/train_onlyseg.py
    unet_checkpoint_dir='../results/bevdiffuser/det/BEVDiffuser_bevdet4d_r50_nocbgs_onlyseg_sam3_metric3d/checkpoint-50000',
    pretrained_model_name_or_path='projects/bevdiffuser/sd21_scheduler',  # local SD-2.1 scheduler
    prediction_type='sample',
    noise_timesteps=100,
    denoise_timesteps=100,
    num_inference_steps=5,
    cfg_scale=2,
    use_classifier_guidence=False)

# frozen BEV target = C0 (EMA weights); the student is NOT initialised from it
load_from = '../results/bevdet/det/bevdet4d-r50-nocbgs_seed0/epoch_24_ema.pth'

runner = dict(type='DiffEpochBasedRunner', max_epochs=24)
find_unused_parameters = False
