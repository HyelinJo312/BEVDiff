# Stage 2 — BEVDiffuser teacher retrained on the segmentation BEV (map segmentation baseline).
# Teacher: results/map_seg/bevdiffuser/BEVDiffuserSeg_tiny_original (Stage 1 config
# bevdiffuser/layout_tiny_mapseg.py). Its x0 comes from the segmentation BEVFormer-tiny
# (--load_from results/map_seg/baseline/bevformer_tiny_seg_v3/epoch_24.pth), which has the
# student's own architecture and input pipeline, so model_target is a copy of the student
# and needs no image conversion. Everything else is bev_tiny_det_semantic_original.py.
_base_ = ['./bev_tiny_det_semantic_original.py']

model_target = None
model_target_img_norm_cfg = None
model_target_img_rescale = 1.0
