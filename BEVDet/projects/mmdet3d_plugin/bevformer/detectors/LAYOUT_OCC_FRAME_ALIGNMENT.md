# Layout teacher reuse in Occ3D

`layout_base_occ.py` uses an independent native detection `BEVFormer` as the
frozen clean target. Its full architecture is specified by `model_target`,
and `model_target_checkpoint` is independent of the student `load_from`.
The launcher retains the previously effective FCOS3D student initialization,
UNet checkpoint, occupancy task/data settings, and 0.5/0.5 task/KD mix.
`UpdateTarget` is not registered, and enabling it is rejected.

The occupancy BEV grid is ego-frame `[-40,-40,-1,40,40,5.4]` at 200×200.
The native detection target and reused diffusion teacher use LiDAR-frame
`[-51.2,-51.2,-5,51.2,51.2,3]` at 200×200. Native LiDAR GT boxes are normalized
using the teacher range; the training object-range filter uses that range too.
Occupancy labels and camera masks remain in their native frame.

The detached native detection BEV enters diffusion **without input resampling**.
Only the frozen diffusion output is transported to the occupancy grid before
KD. At `ego_plane_z=0`, the output transport uses the planar map
`p_lidar = R[:2,:2] p_ego + t[:2] + R[:2,2] z` from each sample's `ego2lidar`.
It samples cell centers with bilinear `grid_sample`, zero padding, and
`align_corners=False`. Full native teacher support is all ones; the strict
full-bilinear-footprint mask excludes padded output cells, and masked KD divides
by valid cells times channels. A sample with less than 1% valid overlap fails.
`mask_invalid=False` uses ordinary full-grid MSE and is not the chosen run.
Missing or invalid `ego2lidar` fails rather than using identity.

This is a planar approximation, not exact 3D registration. The omitted XY
height term is bounded by `||ego2lidar[:2,2]|| * max(|zmin-zplane|,
|zmax-zplane|)`. The helper retains its input-warp API for other uses, but that
path is not invoked here. A CPU geometry check on three unique `LiDAR_TOP`
extrinsics transported a native 200×200 linear XY field from ±51.2 m to the
200×200 occupancy ±40 m grid: all output cells were valid and maximum valid
coordinate error was 1.761e-5 m. This is a geometry-only check, not diffusion
inference. Channels, feature distributions, and the detection-trained target
are not converted.

Stage 1 pairing is recorded in
`results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_base_original_bs4/logs/BEVDiffuser/1789919126.761769/hparams.yml`:
`bev_checkpoint` is the absolute `BEVFormer/ckpts/bevformer_r101_dcn_24ep.pth`
path, `bev_config` is `projects/configs/bevdiffuser/layout_base.py`,
`cfg_options` is null, seed is 0, and `task_loss_scale` is 0.1. A safe
checkpoint-key/shape check found 1089 matching keys, with no missing,
unexpected, or shape-mismatched entries for the native detection model; it did
not load checkpoint tensors into a running model. The selected UNet remains
`results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_base_original_bs4/checkpoint-50000`.
This changes target architecture and feature domain relative to the historical
occupancy-target run; metrics must not be treated as a geometry-only ablation.
The new run name is `DiffBEVFormer_layout_base_occ_det_target_output_aligned`
under the unchanged `../results/diff_bevformer/occ/baseline` parent; no result
is implied.
