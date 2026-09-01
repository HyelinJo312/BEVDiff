import copy

import numpy as np
import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_head

from .bevformer import BEVFormer


@DETECTORS.register_module()
class BEVFormerOcc(BEVFormer):
    """BEVFormer occupancy-only baseline for Occ3D-nuScenes."""

    def __init__(self, *args, occ_head=None, **kwargs):
        super(BEVFormerOcc, self).__init__(*args, **kwargs)
        if occ_head is None:
            raise ValueError('BEVFormerOcc requires occ_head config.')
        self.occ_head = build_head(occ_head)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      gt_occ_semantics=None,
                      gt_occ_mask_camera=None,
                      gt_occ_mask_lidar=None,
                      **kwargs):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.forward_pts_train(
            img_feats,
            img_metas,
            prev_bev=prev_bev,
            gt_occ_semantics=gt_occ_semantics,
            gt_occ_mask_camera=gt_occ_mask_camera,
            gt_occ_mask_lidar=gt_occ_mask_lidar)

    def forward_pts_train(self,
                          pts_feats,
                          img_metas,
                          prev_bev=None,
                          gt_occ_semantics=None,
                          gt_occ_mask_camera=None,
                          gt_occ_mask_lidar=None):
        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        bev_feat = self._bev_embed_to_2d(bev)
        occ_logits = self.occ_head(bev_feat)
        return self.occ_head.loss(
            occ_logits, gt_occ_semantics, gt_occ_mask_camera)

    def _bev_embed_to_2d(self, bev_embed):
        """Convert BEVFormer token layout [B, HW, C] to [B, C, H, W]."""
        if bev_embed.dim() != 3:
            raise ValueError(f'BEV embed must be 3D, got {bev_embed.shape}')

        h, w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        if bev_embed.shape[1] == h * w:
            b, _, c = bev_embed.shape
            return bev_embed.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        if bev_embed.shape[0] == h * w:
            _, b, c = bev_embed.shape
            return bev_embed.permute(1, 2, 0).reshape(b, c, h, w).contiguous()
        raise ValueError(
            f'Cannot infer BEV token dimension from shape {bev_embed.shape} '
            f'with H={h}, W={w}')

    def simple_test(self,
                    img_metas,
                    img=None,
                    prev_bev=None,
                    rescale=False,
                    **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bev = self.pts_bbox_head(
            img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        bev_feat = self._bev_embed_to_2d(bev)
        occ_logits = self.occ_head(bev_feat)
        occ_pred = self.occ_head.get_occ(occ_logits)

        results = []
        for pred in occ_pred:
            pred = pred.detach().cpu().numpy().astype(np.uint8)
            results.append(dict(occ_pred=pred))
        return bev, results
