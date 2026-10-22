"""Diffusion-only adapters for the task path used by tools/dist_train_occ.sh.

The image backbone, temporal BEV extraction and BEV encoder are supplied by
BEVFormer unchanged. Only the modules after the BEV and the occupancy GT
interface are adapted here.
"""

import os
from glob import glob

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.datasets.builder import PIPELINES
from mmdet.models import DETECTORS, HEADS
from mmdet.models.builder import build_loss
from mmdet3d.models.builder import build_head
from torch import nn

from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head_occ import BEVFormerOccHead
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from projects.mmdet3d_plugin.datasets.pipelines.loading_occ import LoadOccGTFromFile


@HEADS.register_module()
class DiffusionOccHead(BEVFormerOccHead):
    """BEV-only task head with the baseline's loss() and get_occ() methods.

    Build only TransformerOcc's use_3d=True decoder and predictor. Inheriting
    the baseline loss/decoding methods keeps class IDs, masking, reduction and
    argmax behavior identical without creating another BEV encoder.
    """

    def __init__(self, in_channels=256, pillar_h=16, out_dim=32,
                 num_classes=18, use_mask=False, loss_occ=None):
        BaseModule.__init__(self)
        if in_channels % pillar_h:
            raise ValueError('in_channels must be divisible by pillar_h')
        self.in_channels = in_channels
        self.pillar_h = pillar_h
        self.num_classes = num_classes
        self.use_mask = use_mask
        self.fp16_enabled = False
        self.loss_occ = build_loss(loss_occ or dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        # Same ConvModule definitions and parameter names as TransformerOcc.
        self.decoder = nn.Sequential(
            ConvModule(in_channels // pillar_h, out_dim, kernel_size=3,
                       stride=1, padding=1, bias=False,
                       conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'),
                       act_cfg=dict(type='ReLU', inplace=True)),
            ConvModule(out_dim, out_dim, kernel_size=3,
                       stride=1, padding=1, bias=False,
                       conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'),
                       act_cfg=dict(type='ReLU', inplace=True)),
        )
        self.predicter = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.Softplus(),
            nn.Linear(out_dim * 2, num_classes))

    def init_weights(self):
        # Match TransformerOcc.init_weights for the task's trainable matrices.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, bev_map):
        """[B,C,Y,X] -> baseline logits [B,X,Y,Z,num_classes]."""
        if bev_map.ndim != 4 or bev_map.shape[1] != self.in_channels:
            raise ValueError(f'Expected [B,{self.in_channels},Y,X], got {tuple(bev_map.shape)}')
        batch_size, _, bev_h, bev_w = bev_map.shape
        volume = bev_map.reshape(batch_size, self.in_channels // self.pillar_h,
                                 self.pillar_h, bev_h, bev_w)
        outputs = self.decoder(volume).permute(0, 4, 3, 2, 1)
        return dict(bev_embed=bev_map, occ=self.predicter(outputs))


@DETECTORS.register_module()
class BEVFormerOccDiffusion(BEVFormer):
    """Keep BEVFormer's feature extraction; apply occupancy loss to a given BEV."""

    def __init__(self, *args, occ_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        if occ_head is None:
            raise ValueError('BEVFormerOccDiffusion requires an occ_head config')
        self.occ_head = build_head(occ_head)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Remap only task weights before the recursive child loads. This works
        # with both MMCV and PyTorch; native diffusion weights take precedence.
        for name in self.occ_head.state_dict():
            source = prefix + 'pts_bbox_head.transformer.' + name
            target = prefix + 'occ_head.' + name
            if source in state_dict:
                state_dict.setdefault(target, state_dict.pop(source))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _bev_embed_to_2d(self, bev):
        height, width = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        channels = self.pts_bbox_head.embed_dims
        if bev.ndim == 4 and tuple(bev.shape[1:]) == (channels, height, width):
            return bev.contiguous()
        if bev.ndim == 3:
            if tuple(bev.shape[1:]) == (height * width, channels):
                return bev.permute(0, 2, 1).reshape(-1, channels, height, width).contiguous()
            if bev.shape[0] == height * width and bev.shape[2] == channels:
                return bev.permute(1, 2, 0).reshape(-1, channels, height, width).contiguous()
        raise ValueError(f'Unexpected BEV shape {tuple(bev.shape)} for C,Y,X={(channels, height, width)}')

    def forward_train(self, given_bev=None, voxel_semantics=None,
                      mask_camera=None, mask_lidar=None, img_metas=None, **kwargs):
        if given_bev is None:
            raise ValueError('The diffusion task path requires given_bev')
        if voxel_semantics is None:
            raise ValueError('The occupancy task loss requires voxel_semantics')
        outputs = self.occ_head(self._bev_embed_to_2d(given_bev))
        expected = outputs['occ'].shape[:-1]
        if voxel_semantics.shape != expected:
            raise ValueError(f'Expected GT [B,X,Y,Z]={tuple(expected)}, got {tuple(voxel_semantics.shape)}')
        return self.occ_head.loss(voxel_semantics, mask_camera, outputs, img_metas=img_metas)

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        outputs = self.occ_head(self._bev_embed_to_2d(bev))
        predictions = self.occ_head.get_occ(outputs, img_metas, rescale=rescale)
        return bev, [dict(occ_pred=pred.detach().cpu().numpy().astype(np.uint8))
                     for pred in predictions]


@PIPELINES.register_module()
class LoadOccGTForDiffusion(LoadOccGTFromFile):
    """Resolve legacy temporal-info tokens, then use the baseline GT loader.

    Images, calibration, temporal queues and semantic/depth conditions retain
    their existing pipeline. All GT arrays stay in native [X,Y,Z] order and
    keep the original 0..17 IDs, including free space (17).
    """

    def __init__(self, occ_root):
        super().__init__(data_root=occ_root)
        self.token_to_label_path = {
            os.path.basename(os.path.dirname(path)): os.path.relpath(path, occ_root)
            for path in glob(os.path.join(occ_root, '*', '*', 'labels.npz'))
        }

    def __call__(self, results):
        if not results.get('load_occ_annotations', True):
            # The existing temporal dataset retains only current-frame GT.
            for key in ('voxel_semantics', 'mask_camera', 'mask_lidar'):
                results[key] = None
            return results
        if 'occ_gt_path' not in results:
            token = results['sample_idx']
            if token not in self.token_to_label_path:
                raise FileNotFoundError(f'Occ3D labels missing for {token} under {self.data_root}')
            results['occ_gt_path'] = self.token_to_label_path[token]
        return super().__call__(results)
