# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from projects.bevdiffuser.ldm.modules.attention import CrossAttention, FeedForward, Normalize


class TemporalBEVWarper(nn.Module):
    def __init__(self, bev_h, bev_w, pc_range):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = tuple(pc_range)

    def forward(self, history_bevs, history_masks, history_lidar2global,
                current_lidar2global, history_valid):
        batch, frames, channels, height, width = history_bevs.shape
        device = history_bevs.device
        dtype = torch.float32

        x = torch.arange(width, device=device, dtype=dtype) + 0.5
        y = torch.arange(height, device=device, dtype=dtype) + 0.5
        x = self.pc_range[0] + x * (
            self.pc_range[3] - self.pc_range[0]) / width
        y = self.pc_range[1] + y * (
            self.pc_range[4] - self.pc_range[1]) / height
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        points = torch.stack((xx, yy, torch.zeros_like(xx),
                              torch.ones_like(xx)), dim=-1)

        source_from_current = torch.linalg.inv(
            history_lidar2global.float()) @ current_lidar2global[:, None].float()
        source_points = torch.einsum(
            'btij,hwj->bthwi', source_from_current, points)

        grid_x = 2 * (source_points[..., 0] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0]) - 1
        grid_y = 2 * (source_points[..., 1] - self.pc_range[1]) / (
            self.pc_range[4] - self.pc_range[1]) - 1
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch * frames, height, width, 2)

        warped_bevs = F.grid_sample(
            history_bevs.reshape(batch * frames, channels, height, width).float(),
            grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        warped_masks = F.grid_sample(
            history_masks.reshape(batch * frames, 1, height, width).float(),
            grid, mode='nearest', padding_mode='zeros', align_corners=False)

        valid = history_valid.reshape(batch * frames, 1, 1, 1).to(warped_bevs)
        warped_bevs = warped_bevs * valid
        warped_masks = warped_masks * valid
        return (
            warped_bevs.reshape(batch, frames, channels, height, width),
            warped_masks.reshape(batch, frames, 1, height, width) > 0.5,
        )


class HistoryBEVEncoder(nn.Module):
    def __init__(self, history_length, feature_channels, num_levels):
        super().__init__()
        input_channels = history_length * (feature_channels + 1)

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, 1),
            nn.GroupNorm(32, feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(32, feature_channels),
            nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 2, stride=2),
                nn.GroupNorm(32, feature_channels),
                nn.SiLU(),
            )
            for _ in range(num_levels - 1)
        ])

    def forward(self, history_bevs, history_masks):
        history = []
        for frame in range(history_bevs.shape[1]):
            mask = history_masks[:, frame].to(history_bevs)
            history.extend((history_bevs[:, frame] * mask, mask))

        feature = self.stem(torch.cat(history, dim=1))
        contexts = {1: feature}
        downsample = 1
        for block in self.down_blocks:
            feature = block(feature)
            downsample *= 2
            contexts[downsample] = feature
        return contexts


class TemporalCrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention(
            query_dim=dim, context_dim=dim, heads=n_heads,
            dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=True)
        self.norm_query = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, x, context):
        x = self.attn(
            self.norm_query(x), context=self.norm_context(context)) + x
        return self.ff(self.norm_ff(x)) + x


class TemporalCrossAttnTransformer(nn.Module):
    def __init__(self, channels, context_channels, n_heads, d_head,
                 spatial_size, depth=1, dropout=0.0):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = Normalize(channels)
        self.context_norm = Normalize(context_channels)
        self.proj_in = nn.Conv2d(channels, inner_dim, 1)
        self.context_proj = nn.Conv2d(context_channels, inner_dim, 1)
        self.blocks = nn.ModuleList([
            TemporalCrossAttnBlock(inner_dim, n_heads, d_head, dropout=dropout) for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(inner_dim, channels, 1)
        self.position = nn.Parameter(
            torch.zeros(1, spatial_size * spatial_size, inner_dim))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x, temporal_context, temporal_keep_mask=None):
        _, _, height, width = x.shape
        residual = x
        query = self.proj_in(self.norm(x))
        context = self.context_proj(self.context_norm(temporal_context))
        query = rearrange(query, 'b c h w -> b (h w) c') + self.position
        context = rearrange(context, 'b c h w -> b (h w) c') + self.position

        for block in self.blocks:
            query = block(query, context)

        output = rearrange(
            query, 'b (h w) c -> b c h w', h=height, w=width)
        output = self.proj_out(output)
        if temporal_keep_mask is not None:
            output = output * temporal_keep_mask[:, None, None, None].to(output)
        return residual + output


def split_temporal_metas(img_metas, frames):
    temporal = []
    for frame in range(frames):
        temporal.append([
            sample if 'filename' in sample else sample[frame]
            for sample in img_metas
        ])
    return temporal


@torch.no_grad()
def extract_temporal_dino(extractor, images, depth_maps, img_metas):
    if images.ndim == 5:
        images = images[:, None]
        depth_maps = depth_maps[:, None]
    frame_metas = split_temporal_metas(img_metas, images.shape[1])
    results = [
        extractor(images[:, frame], frame_metas[frame], depth_maps[:, frame])
        for frame in range(images.shape[1])
    ]
    current = results[-1]
    temporal_bevs = torch.stack([
        result['bev_features'] for result in results], dim=1)
    temporal_masks = torch.stack([
        result['observed_mask'] for result in results], dim=1)
    lidar2global = torch.as_tensor(
        np.asarray([
            [meta['lidar2global'] for meta in sample]
            for sample in zip(*frame_metas)
        ]), device=temporal_bevs.device, dtype=temporal_bevs.dtype)
    return current, temporal_bevs, temporal_masks, lidar2global, frame_metas[-1]


@torch.no_grad()
def build_temporal_dino_condition(extractor, warper, images, depth_maps,
                                  img_metas, temporal_valid):
    current, temporal_bevs, temporal_masks, lidar2global, current_metas = (
        extract_temporal_dino(
            extractor, images, depth_maps, img_metas))
    history_bevs, history_masks = warper(
        temporal_bevs[:, :-1], temporal_masks[:, :-1],
        lidar2global[:, :-1], lidar2global[:, -1], temporal_valid[:, :-1])
    return current, current_metas, history_bevs, history_masks
