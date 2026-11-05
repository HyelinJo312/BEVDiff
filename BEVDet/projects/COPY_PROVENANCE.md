# Provenance of BEVDet/projects

- Copied: 2026-11-04 09:40 KST by Claude Code at user request
- Source: BEVDiff/BEVFormer/projects (working tree, NOT a clean commit)
- BEVDiff HEAD: 51e4c70425c4dbf8c3dc86ff37692a35acf717fe (branch dev2-onlyseg)
- Command: rsync -a --exclude '__pycache__' --exclude '*.pyc' BEVFormer/projects/ BEVDet/projects/
- Verified: diff -rq (excluding __pycache__) reported no differences; 394 files

## Uncommitted state of the source at copy time (git status --short BEVFormer/projects)

```
 M BEVFormer/projects/bevdiffuser/SEMANTIC_BEV_CACHE.md
 M BEVFormer/projects/bevdiffuser/build_semantic_bev_cache.py
 D BEVFormer/projects/bevdiffuser/build_semantic_bev_cache_occ.sh
 M BEVFormer/projects/bevdiffuser/test_only_seg.sh
 M BEVFormer/projects/bevdiffuser/train_only_seg_cache_det.sh
 M BEVFormer/projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ_logcount.py
 D BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_dino_temporal.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_2.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_3.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_4.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_5.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_da3.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_sam.py
 D BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_sam_v2.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_2.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_2_mgd.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_2_mgd_v2.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_da3.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_da3_v2.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_proj.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_sam_mgd.py
 D BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_sam_mgd_v2.py
?? BEVFormer/projects/bevdiffuser/build_semantic_bev_cache.sh
?? BEVFormer/projects/bevdiffuser/train_only_seg_cache_prob.sh
?? BEVFormer/projects/configs/bevdiffuser/bev_base_onlyseg_sam_v2_metric3d.py
?? BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_crosswalk_s08.py
?? BEVFormer/projects/configs/diff_bevformer/bev_base_onlyseg_sam_metric3d.py
```

## Source working-tree diff hash

- sha256 of 'git diff HEAD -- BEVFormer/projects': ee8ffa0df4d213a54afd105eb00e3756a96d8ed01c836cb9d7e593a7a9e4cbcd
