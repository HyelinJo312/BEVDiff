"""Metric-depth semantic lift-and-splat with the v3 BEV encoder interface."""

import numpy as np
import torch
import torch.nn as nn

from .seg_bev_aligner_one_hot_v3 import SegBEVEncoder


class SegBEVAligner(nn.Module):
    """Push observed semantic surfaces into BEV without learned projection.

    Inputs are aligned, padded [B, V, H, W] maps. Depth is in meters;
    ``camera_z`` denotes optical-axis depth, ``ray_distance`` Euclidean range
    from the camera center. ``lidar2img`` must include image augmentation.
    Unknown cells remain zero before the learned embedding/BEV encoder.
    """

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
                 eps=1e-6):
        super().__init__()
        
        if len(pc_range) != 6 or any(pc_range[i] >= pc_range[i + 3] for i in range(3)):
            raise ValueError('pc_range must contain increasing XYZ bounds')
        if len(depth_range) != 2 or not 0 < depth_range[0] < depth_range[1]:
            raise ValueError('depth_range must contain positive increasing meter bounds')
        if depth_type not in ('camera_z', 'ray_distance'):
            raise ValueError('depth_type must be camera_z or ray_distance')
        if splat_mode not in ('nearest', 'bilinear'):
            raise ValueError('splat_mode must be nearest or bilinear')
        if pixel_stride < 1 or int(pixel_stride) != pixel_stride:
            raise ValueError('pixel_stride must be a positive integer')
        if min(bev_h, bev_w) < 5 or eps <= 0 or num_classes < 1:
            raise ValueError('BEV dimensions must be >= 5; eps and num_classes must be positive')
        if len(channel_mult) != 3 or channel_mult[0] != 1:
            raise ValueError('channel_mult must have three entries starting with 1')
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
        # Geometry cache is not a DDP buffer: it must not be broadcast each forward.
        self._pixel_grid = None
        self._pixel_grid_shape = None

        self.prob_to_emb = nn.Sequential(
            nn.Conv2d(num_classes + 1, emb_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(emb_channels, emb_channels, kernel_size=1),
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
    def project_semantics(self, seg_id, img_metas, depth_maps):
        """Return float32 (semantic mass, support) in BEV for inspection.

        Shapes: [B, C, bev_h, bev_w], [B, 1, bev_h, bev_w]. Each observed
        camera contributes one normalized histogram at most. Subpixel support
        below one pixel retains its magnitude; support is sampling mass, not
        a calibrated depth confidence. Unobserved cells have zero mass.
        """
        if depth_maps is None:
            raise ValueError('Push-style splatting requires metric depth_maps')
        if seg_id.ndim != 4 or depth_maps.shape != seg_id.shape:
            raise ValueError('seg_id and depth_maps must have identical [B, V, H, W] shapes')
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
                    inside = torch.isfinite(xyz).all(dim=-1)
                    inside &= ((xyz >= bounds[:3]) & (xyz < bounds[3:])).all(dim=-1)
                    inside_indices = inside.nonzero(as_tuple=True)[0]
                    counts = self._splat(xyz[inside_indices], selected_labels[inside_indices])
                    mass = counts.sum(dim=0, keepdim=True)
                    output[b] += counts / mass.clamp_min(1.0)
                    support[b] += mass
                    view_count[b] += (mass > self.eps).float()
        return output / view_count.clamp_min(1.0), support

    def encode_probabilities(self, probabilities):
        expected = (self.num_classes + 1, self.bev_h, self.bev_w)
        if probabilities.ndim != 4 or tuple(probabilities.shape[1:]) != expected:
            raise ValueError(f'Expected semantic probabilities [B, {expected}]')
        seg_bev = self.prob_to_emb(probabilities.to(self.prob_to_emb[0].weight.dtype))
        return self.seg_bev_encoder(seg_bev + self.bev_pos_embed)

    def forward(self, seg_id, img_metas, depth_maps=None, semantic_probabilities=None):
        if semantic_probabilities is not None:
            if seg_id is not None or depth_maps is not None:
                raise ValueError('Choose cached probabilities or live SAM3/depth, not both')
            return self.encode_probabilities(semantic_probabilities)
        probabilities, _ = self.project_semantics(seg_id, img_metas, depth_maps)
        return self.encode_probabilities(probabilities)
