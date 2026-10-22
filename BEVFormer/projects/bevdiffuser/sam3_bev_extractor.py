# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Metric-depth lift-and-splat of SAM3 feature maps into BEV.

Adapted from ``dinov3_bev_extractor.py``. The geometry is unchanged - unproject
every valid depth pixel, sample the feature map there, splat the vector into the
BEV grid, normalize per camera and average over cameras - but the image frame
the features live in is different, and that is the whole adaptation:

  DINOv3 runs on the dataset's augmented, PADDED frame, so one
  ``feature_image_shape`` describes the whole batch and the patch grid covers the
  padding as well.

  SAM3's ``Sam3Processor`` resizes the raw image to a square model frame, with a
  different scale per axis and no padding, so its feature map covers exactly the
  image CONTENT. Normalizing the sample coordinates therefore uses each view's
  content shape, which ``grid_sample`` handles because it normalizes x and y
  independently. Passing the padded shape here would shift every sample by the
  padding, silently and by up to a few metres at range.

The encoder itself is not here: SAM3 needs its own conda environment, so feature
maps are produced there and handed to this projector, which is plain torch.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _content_shape(meta, view):
    # PadMultiViewImage overwrites img_shape; ori_shape retains resized content.
    pad = meta['pad_shape'][view][:2]
    original = meta.get('ori_shape', meta['img_shape'])[view][:2]
    return original if all(a <= b for a, b in zip(original, pad)) else meta['img_shape'][view][:2]


class SAM3FeatureProjector(nn.Module):
    """Lift valid depth pixels, sample SAM3 features, and splat signed vectors.

    Each camera contributes its weighted mean to a cell. Observed cameras are
    then averaged equally. Unknown cells are exactly zero. No learned projection
    and no class filtering is applied to the features.
    """

    def __init__(self, bev_h=50, bev_w=50, pc_range=(-51.2, -51.2, -5, 51.2, 51.2, 3),
                 depth_range=(2.5, 170), depth_type='camera_z', splat_mode='bilinear',
                 pixel_stride=1, point_chunk_size=8192, eps=1e-6):
        super().__init__()
        if len(pc_range) != 6 or not all(math.isfinite(x) for x in pc_range) or any(pc_range[i] >= pc_range[i + 3] for i in range(3)):
            raise ValueError('pc_range must contain finite increasing XYZ bounds')
        if len(depth_range) != 2 or not 0 < depth_range[0] < depth_range[1]:
            raise ValueError('depth_range must contain positive increasing bounds')
        if depth_type not in ('camera_z', 'ray_distance') or splat_mode not in ('bilinear', 'nearest'):
            raise ValueError('Unsupported depth_type or splat_mode')
        if any(int(x) != x or x < 1 for x in (bev_h, bev_w, pixel_stride, point_chunk_size)) or eps <= 0:
            raise ValueError('Dimensions, strides, chunk size and eps must be positive')
        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.pc_range, self.depth_range = tuple(pc_range), tuple(depth_range)
        self.depth_type, self.splat_mode = depth_type, splat_mode
        self.pixel_stride, self.point_chunk_size = int(pixel_stride), int(point_chunk_size)
        self.eps = eps

    def _splat(self, xyz, features, sums, mass):
        x = (xyz[:, 0] - self.pc_range[0]) * self.bev_w / (self.pc_range[3] - self.pc_range[0])
        y = (xyz[:, 1] - self.pc_range[1]) * self.bev_h / (self.pc_range[4] - self.pc_range[1])
        if self.splat_mode == 'nearest':
            neighbors = [(x.floor().long(), y.floor().long(), torch.ones_like(x))]
        else:
            x, y = x - 0.5, y - 0.5
            x0, y0 = x.floor().long(), y.floor().long()
            fx, fy = x - x0, y - y0
            neighbors = [(x0, y0, (1-fx)*(1-fy)), (x0+1, y0, fx*(1-fy)),
                         (x0, y0+1, (1-fx)*fy), (x0+1, y0+1, fx*fy)]
        for ix, iy, weights in neighbors:
            valid = (ix >= 0) & (ix < self.bev_w) & (iy >= 0) & (iy < self.bev_h)
            index = torch.where(valid, iy * self.bev_w + ix, 0)
            weights = torch.where(valid, weights, 0)
            sums.scatter_add_(1, index[None].expand(features.shape[0], -1), features * weights[None])
            mass.scatter_add_(0, index, weights)

    @torch.no_grad()
    def forward(self, features, depth_maps, img_metas):
        """``features`` [B,V,C,Hf,Wf] covering the image content, depth [B,V,H,W]."""
        if features.ndim != 5 or depth_maps.ndim != 4 or features.shape[:2] != depth_maps.shape[:2]:
            raise ValueError('Expected features [B,V,C,Hf,Wf] and depth [B,V,H,W]')
        bsz, views, channels = features.shape[:3]
        height, width = depth_maps.shape[-2:]
        if len(img_metas) != bsz:
            raise ValueError('Metadata must match the batch')
        device = features.device
        with torch.autocast(device_type=device.type, enabled=False):
            depths = depth_maps.to(device=device, dtype=torch.float32)
            maps = features.float()
            vv, uu = torch.meshgrid(torch.arange(0, height, self.pixel_stride, device=device),
                                    torch.arange(0, width, self.pixel_stride, device=device), indexing='ij')
            pixels = torch.stack((uu, vv, torch.ones_like(uu)), -1).reshape(-1, 3).float()
            cells = self.bev_h * self.bev_w
            output = maps.new_zeros(bsz, channels, cells)
            support = maps.new_zeros(bsz, 1, cells)
            view_count = maps.new_zeros(bsz, 1, cells)
            bounds = maps.new_tensor(self.pc_range)
            for b, meta in enumerate(img_metas):
                matrices = np.asarray(meta['lidar2img'], dtype=np.float32)
                if matrices.shape != (views, 4, 4) or not np.isfinite(matrices).all():
                    raise ValueError('lidar2img must contain V finite 4x4 matrices')
                if any(tuple(s[:2]) != (height, width) for s in meta['pad_shape']):
                    raise ValueError('Depth maps must match the metadata padded image frame')
                inverse = torch.linalg.inv(torch.as_tensor(matrices, device=device))
                for v in range(views):
                    depth = depths[b, v, ::self.pixel_stride, ::self.pixel_stride].reshape(-1)
                    ch, cw = _content_shape(meta, v)
                    valid = torch.isfinite(depth) & (depth >= self.depth_range[0]) & (depth <= self.depth_range[1])
                    valid &= (pixels[:, 0] < cw) & (pixels[:, 1] < ch)
                    selected = valid.nonzero(as_tuple=True)[0]
                    sums, mass = maps.new_zeros(channels, cells), maps.new_zeros(cells)
                    for ids in selected.split(self.point_chunk_size):
                        uv = pixels[ids]
                        rays = uv @ inverse[v, :3, :3].T
                        if self.depth_type == 'ray_distance':
                            rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(self.eps)
                        xyz = rays * depth[ids, None] + inverse[v, :3, 3]
                        inside = torch.isfinite(xyz).all(-1) & ((xyz >= bounds[:3]) & (xyz < bounds[3:])).all(-1)
                        keep = inside.nonzero(as_tuple=True)[0]
                        if keep.numel() == 0:
                            continue
                        uv, xyz = uv[keep], xyz[keep]
                        # The SAM3 map spans the content box, not the padded frame,
                        # so normalize by the content shape of this very view.
                        grid = 2 * (uv[:, :2] + 0.5) / uv.new_tensor([float(cw), float(ch)]) - 1
                        sampled = F.grid_sample(maps[b, v][None], grid[None, None],
                                                mode='bilinear', padding_mode='border',
                                                align_corners=False)[0, :, 0]
                        self._splat(xyz, sampled, sums, mass)
                    observed = mass > self.eps
                    output[b] += (sums / mass.clamp_min(self.eps)[None]) * observed[None]
                    support[b, 0] += mass
                    view_count[b, 0] += observed.float()
            shape = (bsz, 1, self.bev_h, self.bev_w)
            return dict(bev_features=(output / view_count.clamp_min(1)).reshape(bsz, channels, self.bev_h, self.bev_w),
                        support=support.reshape(shape), observed_mask=(view_count > 0).reshape(shape))


class SAM3BEVExtractor(nn.Module):
    """Project several SAM3 taps through one shared geometry.

    The taps differ only in channel count and resolution, so every one of them
    is lifted with the same depth, calibration and splat; differences between
    the resulting BEV maps come from the features alone.
    """

    def __init__(self, projector=None):
        super().__init__()
        self.projector = projector if isinstance(projector, nn.Module) else SAM3FeatureProjector(**(projector or {}))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, tap_features, depth_maps, img_metas):
        """``tap_features`` maps a tap name to its [B,V,C,Hf,Wf] feature stack."""
        return {name: self.projector(features, depth_maps, img_metas)
                for name, features in tap_features.items()}
