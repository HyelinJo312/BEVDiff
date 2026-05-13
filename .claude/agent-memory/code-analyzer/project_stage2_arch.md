---
name: Stage 2 architecture map (BEVDiffV2)
description: Where each piece of the Stage-2 distillation pipeline lives, including the seg variant. Use to navigate quickly without re-greping.
type: project
---

Stage-2 distillation pipeline files (verified 2026-05-10):

- Detector (baseline): `BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer.py` — class `DiffBEVFormer`.
- Detector (ours seg): `BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer_seg.py` — classes `DiffBEVFormerSeg` and `DiffBEVFormerSegV2`.
- Train entry baseline: `BEVFormer/tools/train.py` (imports `bevdiffuser.BEVDiffuser`).
- Train entry seg: `BEVFormer/tools/train_seg.py` (imports `bevdiffuser_seg.BEVDiffuser`).
- BEVDiffuser wrapper baseline: `BEVFormer/tools/bevdiffuser.py` — `forward(x, condition, grad_fn)` returns `x`.
- BEVDiffuser wrapper seg: `BEVFormer/tools/bevdiffuser_seg.py` — `forward(x, img_metas, condition, segmaps, depth_maps, grad_fn)` returns `(x, seg_bev_prob)`. seg_bev_prob is captured only on the LAST denoise step (and only from the conditional branch).
- UNet (seg v4): `BEVFormer/projects/bevdiffuser/layout_diffusion/layout_seg_diffusion_unet_v4.py` — imports `seg_bev_aligner_one_hot_da3.SegBEVAligner`.
- Seg aligner: `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_bev_aligner_one_hot_da3.py` — exposes `forward()` and `compute_prob_only()`. `seg_bev_prob` is GT-derived (deterministic given seg_id), not learnt.
- Runner: `BEVFormer/projects/mmdet3d_plugin/bevformer/runner/diff_epoch_based_runner.py` — sets `bev_diffuser.eval()` and `model_target.eval()` once at init. Passes `progress = (epoch+1)/max_epochs`.
- Hook: `UpdateTarget` in `BEVFormer/projects/mmdet3d_plugin/bevformer/hooks/custom_hooks.py` — copies student → teacher each iter when `iter_interval=0` (with `every_n_iters` semantics this fires every iter).
- Active seg config (per `dist_train_seg.sh`): `BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_adapter.py` — `noise_timesteps=denoise_timesteps=100`, `num_inference_steps=5`, `use_proj=True`, `use_aux_seg=False`, `use_bev_rel=False`.
- Active baseline config: `BEVFormer/projects/configs/diff_bevformer/layout_tiny.py` — `noise_timesteps=denoise_timesteps=5`.
- Stage-1 trainer (seg): `BEVFormer/projects/bevdiffuser/train_bev_diffuser_seg_v2.py` — samples timesteps uniformly in [0, 1000) (SD scheduler default `num_train_timesteps`), `prediction_type="sample"`. Latents are raw BEVFormer-tiny `only_bev=True` features (no scaling).

Key convention: any loss key whose name contains `'bev'` is routed into `bev_loss` bucket by `_parse_losses_mix` and combined with task_loss as `(1-w)*task + w*bev`, w=0.5.
