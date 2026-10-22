# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Frozen DINOv3 patch/CLS extraction and metric-depth feature lift-and-splat.

Inputs use the dataset's augmented image frame; no extra resize/crop is applied.
The spatial target is BCHW, while CLS stays [batch, camera, hidden_size].
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


class FrozenDINOv3Encoder(nn.Module):
    """Load HF weights once, and extract normalized final patch and CLS tokens.

    ``model`` is an injection point for offline tests. Production loads both
    AutoModel and the checkpoint's image normalization settings from HF.
    Requires a Transformers release supporting DINOv3 (e.g. 4.56.2).
    """

    def __init__(self, model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
                 revision='main', cache_dir=None, local_files_only=False,
                 view_batch_size=2, inference_dtype='float32', model=None,
                 image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)):
        super().__init__()
        if view_batch_size < 1 or int(view_batch_size) != view_batch_size:
            raise ValueError('view_batch_size must be a positive integer')
        if inference_dtype not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('inference_dtype must be float32, float16 or bfloat16')
        if model is None:
            from transformers import AutoImageProcessor, AutoModel
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            if 'dinov3_vit' not in CONFIG_MAPPING:
                raise ImportError('DINOv3 requires a newer Transformers release; '
                                  'use a separate environment with transformers==4.56.2.')
            options = dict(revision=revision, cache_dir=cache_dir,
                           local_files_only=local_files_only)
            processor = AutoImageProcessor.from_pretrained(model_name, **options)
            image_mean, image_std = processor.image_mean, processor.image_std
            model = AutoModel.from_pretrained(model_name, **options)
        self.model = model.requires_grad_(False).eval()
        self.hidden_size = int(model.config.hidden_size)
        self.patch_size = int(model.config.patch_size)
        self.num_register_tokens = int(model.config.num_register_tokens)
        self.view_batch_size = int(view_batch_size)
        self.inference_dtype = inference_dtype
        self.model_name = model_name
        self.revision = getattr(model.config, '_commit_hash', None) or revision
        self.register_buffer('image_mean', torch.tensor(image_mean).reshape(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).reshape(1, 3, 1, 1))
        self.train(False)

    def train(self, mode=True):
        # Remain frozen/eval even when an enclosing module calls train().
        return super().train(False)

    def preprocess(self, images, img_metas):
        if images.ndim != 5 or images.shape[2] != 3 or len(img_metas) != images.shape[0]:
            raise ValueError('images must be [B, V, 3, H, W], with B metadata entries')
        _, views, _, height, width = images.shape
        normalized = []
        for b, meta in enumerate(img_metas):
            if len(meta['pad_shape']) != views or any(tuple(s[:2]) != (height, width) for s in meta['pad_shape']):
                raise ValueError('images must match the metadata padded frame')
            norm = meta['img_norm_cfg']
            mean = images.new_tensor(norm['mean'], dtype=torch.float32).view(1, 3, 1, 1)
            std = images.new_tensor(norm['std'], dtype=torch.float32).view(1, 3, 1, 1)
            rgb = images[b].float() * std + mean
            if not norm.get('to_rgb', True):
                rgb = rgb[:, [2, 1, 0]]
            rgb = rgb.clamp(0, 255) / 255.0
            values = (rgb - self.image_mean) / self.image_std
            for v in range(views):
                h, w = map(int, _content_shape(meta, v))
                values[v, :, h:, :] = 0
                values[v, :, :, w:] = 0
            normalized.append(values)
        pixels = torch.cat(normalized)
        # Padding preserves the projection; normalized zero is the DINO mean.
        pixels = F.pad(pixels, (0, -width % self.patch_size, 0, -height % self.patch_size))
        return pixels

    @torch.no_grad()
    def forward(self, images, img_metas):
        pixels = self.preprocess(images, img_metas)
        b, v = images.shape[:2]
        h, w = pixels.shape[-2:]
        ph, pw = h // self.patch_size, w // self.patch_size
        patch_parts, cls_parts = [], []
        use_amp = self.inference_dtype != 'float32'
        dtype = getattr(torch, self.inference_dtype) if use_amp else torch.bfloat16
        with torch.autocast(device_type=pixels.device.type, dtype=dtype, enabled=use_amp):
            for chunk in pixels.split(self.view_batch_size):
                tokens = self.model(pixel_values=chunk, return_dict=True).last_hidden_state
                expected = (chunk.shape[0], 1 + self.num_register_tokens + ph * pw, self.hidden_size)
                if tuple(tokens.shape) != expected:
                    raise ValueError(f'Unexpected DINOv3 token shape {tuple(tokens.shape)}; expected {expected}')
                cls_parts.append(tokens[:, 0].float())
                patch_parts.append(tokens[:, 1 + self.num_register_tokens:].float())
        patches = torch.cat(patch_parts).transpose(1, 2).reshape(b, v, self.hidden_size, ph, pw)
        return dict(patch_features=patches, cls_tokens=torch.cat(cls_parts).reshape(b, v, self.hidden_size),
                    feature_image_shape=(h, w))


class MetricDepthFeatureProjector(nn.Module):
    """Lift valid depth pixels, sample patch features, and splat signed vectors.

    Each camera contributes its weighted mean to a cell. Observed cameras are
    then averaged equally. Unknown cells are exactly zero. No learned target
    projection or semantic-class filtering is applied to the DINO features.
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
    def forward(self, patch_features, depth_maps, img_metas, feature_image_shape):
        if patch_features.ndim != 5 or depth_maps.ndim != 4 or patch_features.shape[:2] != depth_maps.shape[:2]:
            raise ValueError('Expected patch features [B,V,C,Hp,Wp] and depth [B,V,H,W]')
        bsz, views, channels = patch_features.shape[:3]
        height, width = depth_maps.shape[-2:]
        feature_h, feature_w = feature_image_shape
        if feature_h < height or feature_w < width or len(img_metas) != bsz:
            raise ValueError('Feature frame must include the depth frame; metadata must match batch')
        device = patch_features.device
        with torch.autocast(device_type=device.type, enabled=False):
            depths = depth_maps.to(device=device, dtype=torch.float32)
            patches = patch_features.float()
            vv, uu = torch.meshgrid(torch.arange(0, height, self.pixel_stride, device=device),
                                    torch.arange(0, width, self.pixel_stride, device=device), indexing='ij')
            pixels = torch.stack((uu, vv, torch.ones_like(uu)), -1).reshape(-1, 3).float()
            cells = self.bev_h * self.bev_w
            output = patches.new_zeros(bsz, channels, cells)
            support = patches.new_zeros(bsz, 1, cells)
            view_count = patches.new_zeros(bsz, 1, cells)
            bounds = patches.new_tensor(self.pc_range)
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
                    sums, mass = patches.new_zeros(channels, cells), patches.new_zeros(cells)
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
                        # align_corners=False: patch centers correspond to image
                        # coordinates (patch_size-1)/2, not to pixel zero.
                        grid = 2 * (uv[:, :2] + 0.5) / uv.new_tensor([feature_w, feature_h]) - 1
                        features = F.grid_sample(patches[b, v][None], grid[None, None],
                                                 mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0]
                        self._splat(xyz, features, sums, mass)
                    observed = mass > self.eps
                    output[b] += (sums / mass.clamp_min(self.eps)[None]) * observed[None]
                    support[b, 0] += mass
                    view_count[b, 0] += observed.float()
            shape = (bsz, 1, self.bev_h, self.bev_w)
            return dict(bev_features=(output / view_count.clamp_min(1)).reshape(bsz, channels, self.bev_h, self.bev_w),
                        support=support.reshape(shape), observed_mask=(view_count > 0).reshape(shape))


class DINOv3BEVExtractor(nn.Module):
    """Frozen image-to-BEV target provider; CLS is retained without conditioning."""

    def __init__(self, encoder=None, projector=None):
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.Module) else FrozenDINOv3Encoder(**(encoder or {}))
        self.projector = projector if isinstance(projector, nn.Module) else MetricDepthFeatureProjector(**(projector or {}))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, img_metas, depth_maps):
        encoded = self.encoder(images, img_metas)
        projected = self.projector(encoded['patch_features'], depth_maps, img_metas, encoded['feature_image_shape'])
        projected['cls_tokens'] = encoded['cls_tokens']
        return projected
