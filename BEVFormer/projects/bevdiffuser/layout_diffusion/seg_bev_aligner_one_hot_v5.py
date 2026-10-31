# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from LayoutDiffusion
#   (https://github.com/ZGCTroy/LayoutDiffusion)
# Copyright (c) 2023 LayoutDiffusion authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


"""Metric-depth semantic lift-and-splat with the v3 BEV encoder interface."""

import math
import numpy as np
import torch
import torch.nn as nn

from .seg_bev_aligner_one_hot_v3 import SegBEVEncoder


# 'share': per-class pixel share (sums to <= 1 per cell; the legacy prior).
# 'share+logcount': share concatenated with log1p(raw per-class pixel count) /
#   log1p(logcount_scale). The count preserves raw mass normalized away by share.
SEMANTIC_TYPES = ('share', 'share+logcount')


def semantic_input(share, counts=None, semantic_type='share', logcount_scale=1e5):
    if semantic_type == 'share':
        return share
    if semantic_type != 'share+logcount':
        raise ValueError(f'semantic_type must be one of {SEMANTIC_TYPES}, got {semantic_type!r}')
    if counts is None:
        raise ValueError("semantic_type='share+logcount' requires raw per-class counts")
    if counts.shape != share.shape:
        raise ValueError(f'counts {tuple(counts.shape)} must match share {tuple(share.shape)}')
    scale = math.log1p(logcount_scale)
    if isinstance(share, np.ndarray):
        logcount = np.log1p(np.clip(counts.astype(np.float32), 0, None)) / scale
        return np.concatenate([share, logcount], axis=-3).astype(np.float32)
    logcount = torch.log1p(counts.float().clamp_min(0)) / scale
    return torch.cat([share, logcount.to(share.dtype)], dim=-3)


class GlobalSemEncoder(nn.Module):
    """Encode a spatial semantic BEV feature into a global vector."""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, channels) * channels ** -0.5)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
        )

    def forward(self, feature):
        tokens = feature.flatten(2).transpose(1, 2)
        query = self.query.expand(feature.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.proj(pooled[:, 0])


class SegBEVAligner(nn.Module):
    def __init__(self, bev_h=64, bev_w=64,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 num_classes=16, 
                 emb_channels=256, 
                 channel_mult=(1, 2, 4),
                 sky_as_ignore=True, 
                 depth_range=(2.5, 170.0),
                 depth_type='camera_z', 
                 splat_mode='bilinear',
                 pixel_stride=1, 
                 eps=1e-6,
                 output_frame='lidar',
                 use_log_share=False,
                 log_share_eps=0.01,
                 semantic_type='share',
                 logcount_scale=1e5,
                 use_legacy_prob_embed=False):
        super().__init__()
        if semantic_type not in SEMANTIC_TYPES:
            raise ValueError(f'semantic_type must be one of {SEMANTIC_TYPES}, got {semantic_type!r}')
        if not math.isfinite(logcount_scale) or logcount_scale <= 0:
            raise ValueError('logcount_scale must be positive and finite')
        if len(pc_range) != 6 or any(pc_range[i] >= pc_range[i + 3] for i in range(3)):
            raise ValueError('pc_range must contain increasing XYZ bounds')
        if len(depth_range) != 2 or not 0 < depth_range[0] < depth_range[1]:
            raise ValueError('depth_range must contain positive increasing meter bounds')

        self.bev_h, self.bev_w = bev_h, bev_w
        self.pc_range = tuple(pc_range)
        self.num_classes = num_classes
        self.emb_channels = emb_channels
        self.sky_id = num_classes if sky_as_ignore else None
        self.depth_range = tuple(depth_range)
        self.depth_type = depth_type
        self.splat_mode = splat_mode
        self.pixel_stride = int(pixel_stride)
        self.eps = eps
        self.output_frame = output_frame
        self.use_log_share = use_log_share
        self.log_share_eps = log_share_eps
        self.semantic_type = semantic_type
        self.logcount_scale = logcount_scale
        self.input_channels = (num_classes + 1) * (2 if semantic_type == 'share+logcount' else 1)
        # Geometry cache is not a DDP buffer: it must not be broadcast each forward.
        self._pixel_grid = None
        self._pixel_grid_shape = None

        if use_legacy_prob_embed:
            self.prob_to_emb = nn.Sequential(
                nn.Conv2d(self.input_channels, emb_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(emb_channels, emb_channels, kernel_size=1),
            )
        else:
            self.prob_to_emb = nn.Sequential(
                nn.Conv2d(self.input_channels, emb_channels, kernel_size=1),
                nn.SiLU(inplace=True),
            )
        self.bev_pos_embed = nn.Parameter(torch.zeros(1, emb_channels, bev_h, bev_w))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)
        self.seg_bev_encoder = SegBEVEncoder(emb_channels, channel_mult=channel_mult)

    def _splat(self, xyz, labels):
        """Accumulate class mass without allocating a per-pixel one-hot tensor."""
        cells = self.bev_h * self.bev_w
        counts = xyz.new_zeros((self.num_classes + 1) * cells)
        x = (xyz[:, 0] - self.pc_range[0]) * self.bev_w / (self.pc_range[3] - self.pc_range[0])
        y = (xyz[:, 1] - self.pc_range[1]) * self.bev_h / (self.pc_range[4] - self.pc_range[1])
        if self.splat_mode == 'nearest':
            neighbors = [(x.floor().long(), y.floor().long(), torch.ones_like(x))]
        else:
            # Integer coordinates denote cell centers, not lower cell corners.
            x, y = x - 0.5, y - 0.5
            x0, y0 = x.floor().long(), y.floor().long()
            fx, fy = x - x0, y - y0
            neighbors = [(x0, y0, (1 - fx) * (1 - fy)),
                         (x0 + 1, y0, fx * (1 - fy)),
                         (x0, y0 + 1, (1 - fx) * fy),
                         (x0 + 1, y0 + 1, fx * fy)]
        for ix, iy, weight in neighbors:
            valid = (ix >= 0) & (ix < self.bev_w) & (iy >= 0) & (iy < self.bev_h)
            # Fixed-size indexing avoids four CUDA nonzero synchronizations per neighbor.
            index = labels * cells + iy * self.bev_w + ix
            index = torch.where(valid, index, torch.zeros_like(index))
            counts.scatter_add_(0, index, torch.where(valid, weight, torch.zeros_like(weight)))
        return counts.view(self.num_classes + 1, self.bev_h, self.bev_w)

    @torch.no_grad()
    def project_semantics(self, seg_id, img_metas, depth_maps, return_counts=False):
        """Return float32 (semantic mass, support) in BEV for inspection.
        """
        B, V, H, W = seg_id.shape
        if len(img_metas) != B:
            raise ValueError('img_metas batch size does not match the maps')
        device = seg_id.device
        depth_maps = depth_maps.to(device=device, dtype=torch.float32)
        stride = self.pixel_stride
        if self._pixel_grid_shape != (H, W, stride) or self._pixel_grid.device != device:
            vv, uu = torch.meshgrid(torch.arange(0, H, stride, device=device),
                                    torch.arange(0, W, stride, device=device), indexing='ij')
            self._pixel_grid = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1).float().reshape(-1, 3)
            self._pixel_grid_shape = (H, W, stride)
        pixels = self._pixel_grid
        output = torch.zeros(B, self.num_classes + 1, self.bev_h, self.bev_w, device=device)
        support = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        view_count = torch.zeros_like(support)
        class_counts = torch.zeros_like(output) if return_counts else None
        bounds = pixels.new_tensor(self.pc_range)

        # Keep inverses and geometric products in float32 under training AMP.
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16 if device.type == 'cpu' else torch.float16,
                            enabled=False):
            for b, meta in enumerate(img_metas):
                projection_array = np.asarray(meta['lidar2img'], dtype=np.float32)
                
                if projection_array.shape != (V, 4, 4) or not np.isfinite(projection_array).all():
                    raise ValueError('lidar2img must contain V finite 4x4 projection matrices')
                projections = torch.as_tensor(projection_array, dtype=torch.float32, device=device)

                if self.output_frame == 'ego':
                    if 'ego2lidar' not in meta:
                        raise ValueError('ego output requires ego2lidar in img_metas')
                    calibration = np.asarray(meta['ego2lidar'], dtype=np.float32)
                    if calibration.shape != (4, 4) or not np.isfinite(calibration).all():
                        raise ValueError('ego2lidar must be a finite 4x4 matrix')
                    ego2lidar = torch.as_tensor(calibration, dtype=torch.float32, device=device)
                    lidar2ego = torch.linalg.inv(ego2lidar)
                
                if 'pad_shape' in meta and any(tuple(s[:2]) != (H, W) for s in meta['pad_shape']):
                    raise ValueError('Maps must already match the padded image frame')
                inverse = torch.linalg.inv(projections)
                
                for v in range(V):
                    depth = depth_maps[b, v, ::stride, ::stride].reshape(-1)
                    labels = seg_id[b, v, ::stride, ::stride].reshape(-1)
                    valid = torch.isfinite(depth) & (depth >= self.depth_range[0]) & (depth <= self.depth_range[1])
                    # IDs 0 and -1 denote unknown; never treat them as a surface class.
                    valid &= (labels > 0) & (labels <= self.num_classes) & (labels == labels.long())
                    
                    if self.sky_id is not None:
                        valid &= labels != self.sky_id
                    # Compact once and reuse indices; boolean indexing otherwise repeats nonzero.
                    selected = valid.nonzero(as_tuple=True)[0]
                    
                    if selected.numel() == 0:
                        continue
                    selected_labels = labels[selected].long()
                    rays = pixels[selected] @ inverse[v, :3, :3].T
                    
                    if self.depth_type == 'ray_distance':
                        rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(self.eps)
                        
                    xyz = rays * depth[selected, None] + inverse[v, :3, 3]
                    if self.output_frame == 'ego':
                        xyz = xyz @ lidar2ego[:3, :3].T + lidar2ego[:3, 3]
                    inside = torch.isfinite(xyz).all(dim=-1)
                    inside &= ((xyz >= bounds[:3]) & (xyz < bounds[3:])).all(dim=-1)
                    inside_indices = inside.nonzero(as_tuple=True)[0]
                    counts = self._splat(xyz[inside_indices], selected_labels[inside_indices])
                    mass = counts.sum(dim=0, keepdim=True)
                    output[b] += counts / mass.clamp_min(1.0)
                    support[b] += mass
                    view_count[b] += (mass > self.eps).float()
                    if return_counts:
                        class_counts[b] += counts
        share = output / view_count.clamp_min(1.0)
        if return_counts:
            return share, support, class_counts
        return share, support

    def encode_prob(self, probabilities):
        """Encode the semantic input ([share] or [share, logcount]).
        """
        if probabilities.ndim != 4 or probabilities.shape[1] != self.input_channels:
            raise ValueError(
                f'semantic_type={self.semantic_type!r} expects {self.input_channels} input channels, '
                f'got {tuple(probabilities.shape)}; dataset and aligner semantic_type must match')
        if self.use_log_share:
            classes = self.num_classes + 1
            share = torch.log1p(probabilities[:, :classes].float() / self.log_share_eps)
            share = share / math.log1p(1.0 / self.log_share_eps)
            probabilities = torch.cat([share, probabilities[:, classes:].float()], dim=1)
            
        seg_bev = self.prob_to_emb(probabilities.to(self.prob_to_emb[0].weight.dtype))
        return self.seg_bev_encoder(seg_bev + self.bev_pos_embed)

    def forward(self, seg_id, img_metas, depth_maps=None, semantic_prob=None):
        if semantic_prob is not None:
            return self.encode_prob(semantic_prob)

        if self.semantic_type == 'share':
            prob, _ = self.project_semantics(seg_id, img_metas, depth_maps)
            return self.encode_prob(prob)

        prob, _, counts = self.project_semantics(seg_id, img_metas, depth_maps, return_counts=True)
        return self.encode_prob(semantic_input(prob, counts, self.semantic_type, self.logcount_scale))
