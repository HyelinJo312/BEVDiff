import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32


class SegEmbedEncoder(nn.Module):
    """
    Lightweight pyramid encoder:
      1. Embedding(64-dim) → full res에서 3x3 conv (boundary context, 가벼움)
      2. Stride-2 conv → 절반 해상도로 축소 + out_channels 확장
      3. 축소된 해상도에서 추가 3x3 conv (context 확장)

    """
    def __init__(self, num_classes, embed_dim, out_channels):
        super().__init__()
        self.embed = nn.Embedding(num_classes + 1, embed_dim)
        self.downsample_factor = 2

        self.encoder = nn.Sequential(
            # Stage 1: full res, 경량 채널 → boundary context 추출
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            # Stage 2: stride-2 downsample + 채널 확장
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            # Stage 3: half res, full 채널 → context 정제
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, seg_id):
        B, V, H, W = seg_id.shape
        seg_id = seg_id.view(B * V, H, W)

        seg_id = seg_id.clamp(min=-1, max=16)
        seg_id = torch.where(seg_id == -1, torch.zeros_like(seg_id), seg_id)

        seg_emb = self.embed(seg_id.long())               # [B*V, H, W, embed_dim]
        seg_emb = seg_emb.permute(0, 3, 1, 2).contiguous()  # [B*V, embed_dim, H, W]
        seg_emb = self.encoder(seg_emb)                   # [B*V, out_channels, H//2, W//2]

        return seg_emb


class SegBEVEncoder(nn.Module):
    """
    Input:  [B, C_in, H, W]
    Output: {1: [B, C1, H, W], 2: [B, C2, H//2, W//2], 4: [B, C4, H//4, W//4]}

    """
    def __init__(self, in_channels, channel_mult=(1, 2, 4)):
        super().__init__()
        c1 = in_channels * channel_mult[0]
        c2 = in_channels * channel_mult[1]
        c4 = in_channels * channel_mult[2]

        # v2: stride-2 Conv (learnable spatial mixing) + SiLU
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c4, kernel_size=2, stride=2),
            nn.SiLU(),
        )

    def forward(self, bev_ctx):
        """
        Args:
            bev_ctx: (B, C, bevH, bevW)
        Returns:
            dict {1: (B, C1, H1, W1), 2: (B, C2, H2, W2), 4: (B, C4, H4, W4)}
        """
        s1 = bev_ctx                # (B, 128, 50, 50)
        s2 = self.down1(s1)         # (B, 256, 25, 25)
        s4 = self.down2(s2)         # (B, 512, 12, 12)

        return {
            1: s1,
            2: s2,
            4: s4,
        }


class SegBEVAligner(nn.Module):
    """
    BEVFormer-style IPM (Inverse Perspective Mapping) for multi-view
    segmentation ID map -> BEV feature, modeled after DINOBevAligner.

    v2 changes vs v1:
      1. SegEmbedEncoder: 3x3 conv proj 활성화 (boundary context extraction)
      2. grid_sample: mode='nearest' -> mode='bilinear' (연속 feature + gradient flow)
      3. SegBEVEncoder: AvgPool+1x1 -> stride-2 Conv (learnable spatial mixing)

    Inputs:
      - seg_id:    (B, V, H, W) segmentation class IDs
      - img_metas: list of dicts (len=B), each with 'lidar2img': (V, 4, 4)
    Returns:
      - dict {1: ..., 2: ..., 4: ...} multi-scale BEV features
    """

    def __init__(
        self,
        bev_h=64,
        bev_w=64,
        cam_view=6,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        num_classes=16,
        embed_dim=32,
        emb_channels=64,
        final_dim=(480, 800),       
        channel_mult=[1,2,4],
        return_multiscale=False,
        eps=1e-6,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.emb_channels = emb_channels
        self.final_dim = final_dim
        self.return_multiscale = return_multiscale
        self.eps = eps

        # ---- Segmentation embedding encoder (full resolution, boundary-aware) ----
        self.seg_encoder = SegEmbedEncoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            out_channels=self.emb_channels,
        )

        # ---- Feature normalization: BEV 공간 feature map 단위 정규화 ----
        # self.post_norm = nn.GroupNorm(num_groups=8, num_channels=self.emb_channels)

        self._w_view = nn.Parameter(th.zeros(1, cam_view, 1))

        self.bev_pos_embed = nn.Parameter(th.zeros(1, self.emb_channels, bev_h, bev_w))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        if self.return_multiscale:
            self.seg_bev_encoder = SegBEVEncoder(
                in_channels=self.emb_channels,
                channel_mult=channel_mult,
            )

    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4,
                              dim='3d', bs=1, device='cuda', dtype=th.float32):
        """Identical to DINOBevAligner._get_reference_points."""
        if dim == '3d':
            zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar,
                             dtype=dtype, device=device
                             ).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = th.linspace(0.5, W - 0.5, W,
                             dtype=dtype, device=device
                             ).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = th.linspace(0.5, H - 0.5, H,
                             dtype=dtype, device=device
                             ).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = th.stack((xs, ys, zs), -1)  # (D, H, W, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (D, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # (bs, D, H*W, 3)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = th.meshgrid(
                th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = th.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)  # (bs, H*W, 1, 2)
            return ref_2d
        else:
            raise ValueError("dim must be '3d' or '2d'")

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
        lidar2img = th.stack([
            th.as_tensor(np.array(m['lidar2img']), dtype=th.float32, device=reference_points.device)
            for m in img_metas
        ])

        pc_range = self.pc_range
        ref = reference_points.clone()
        ref[..., 0:1] = ref[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref = th.cat((ref, th.ones_like(ref[..., :1])), -1)            # (bs, D, Q, 4)

        ref = ref.permute(1, 0, 2, 3)                                  # (D, B, Q, 4)
        D, B, Q = ref.size()[:3]
        num_cam = lidar2img.size(1)

        ref = ref.view(D, B, 1, Q, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # (D,B,N,Q,4,1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, Q, 1, 1)

        cam = th.matmul(lidar2img.to(th.float32), ref.to(th.float32)).squeeze(-1)  # (D,B,N,Q,4)
        eps = 1e-5
        depth = cam[..., 2:3]
        bev_mask = (depth > eps)                                                   # (D,B,N,Q,1)

        uv = cam[..., 0:2] / th.maximum(depth, th.ones_like(depth) * eps)          # (D,B,N,Q,2)

        # (V,B,Q,D,2), (V,B,Q,D)
        uv = uv.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()
        return uv, bev_mask

    def forward(self, seg_id, img_metas):
        assert seg_id.dim() == 4, f'seg_id is {seg_id.shape}'
        B, V, H, W = seg_id.shape
        device = seg_id.device
        C = self.emb_channels

        # (1) Embed segmentation IDs -> feature maps
        seg_emb = self.seg_encoder(seg_id)  # [B*V, C, fH, fW]

        # (2) Generate BEV 3D reference points
        Z_bins = int(round(self.pc_range[5] - self.pc_range[2]))
        ref_3d = self._get_reference_points(
            self.bev_h, self.bev_w, Z=Z_bins,
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d', bs=B, device=device, dtype=seg_emb.dtype
        )

        # (3) Project 3D refs to 2D image coords
        uv, bev_mask = self.point_sampling(ref_3d, img_metas)  # (V,B,Q,D,2), (V,B,Q,D)

        # (4) Pixel coords → normalised coords for grid_sample
        Q = self.bev_h * self.bev_w
        QD = Q * self.num_points_in_pillar
        
        imgH, imgW = self.final_dim  # original image size (480, 800)
        ds = self.seg_encoder.downsample_factor
        fH, fW = imgH // ds, imgW // ds  # feature map size (240, 400)

        # Reshape uv: (V,B,Q,D,2) → (B,V,QD,2)
        uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)

        u = uv_flat[..., 0]  # (B, V, QD) — pixel coords in original image
        v = uv_flat[..., 1]

        # Validity check in original image space (before downscale)
        valid_in = (u >= 0) & (u <= (imgW - 1)) & (v >= 0) & (v <= (imgH - 1))
        bev_mask_flat = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD)
        mask_bv = bev_mask_flat & valid_in  # (B, V, QD)

        # Scale to feature map space, then normalise to [-1, 1]
        u_feat = u / ds  # pixel coords in feature map space
        v_feat = v / ds
        gx = 2.0 * (u_feat / (fW - 1.0)) - 1.0
        gy = 2.0 * (v_feat / (fH - 1.0)) - 1.0
        grid = th.stack([gx, gy], dim=-1)  # (B, V, QD, 2)

        # (5) Bilinear sampling from seg feature maps
        grid_v = grid.view(B * V, QD, 1, 2)

        # v2: mode='bilinear' (연속 embedding feature에 적합, gradient flow 활성화)
        sampled = F.grid_sample(
            seg_emb, grid_v, mode='bilinear',
                padding_mode='border', align_corners=True
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B*V, Q*D, C)
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, C)

        # (7) Pillar mean + view-weighted mean
        # mask_bv: (B, V, Q*D) -> (B, V, Q, D, 1)
        mask = mask_bv.view(B, V, Q, self.num_points_in_pillar).unsqueeze(-1).float()  # (B,V,Q,D,1)
        sampled = sampled * mask
        denom_D = mask.sum(dim=3, keepdim=True).clamp_min(self.eps)  # (B,V,Q,1,1)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D  # (B,V,Q,1,C)
        feat_v = feat_v.squeeze(3)  # (B,V,Q,C)

        # View-weighted mean
        w = F.softplus(self._w_view).expand(B, -1, -1)  # (B, V, 1)
        w = w.unsqueeze(-1)  # (B, V, 1, 1)
        view_valid = (denom_D.squeeze(3) > 0).float()  # (B, V, Q, 1)
        num = (feat_v * w).sum(dim=1)  # (B, Q, C)
        den = (w * view_valid).sum(dim=1).clamp_min(self.eps)  # (B, Q, 1)
        f_bev = num / den  # (B, Q, C)

        # (8) Reshape to (B, C, bev_h, bev_w)
        seg_bev = f_bev.permute(0, 2, 1).contiguous().view(B, C, self.bev_h, self.bev_w)

        # (6) GroupNorm: reshape 후 BEV 공간 map 단위로 정규화
        # seg_bev = self.post_norm(seg_bev)

        # Add BEV positional embedding
        seg_bev = seg_bev + self.bev_pos_embed

        # (9) Multi-scale encoding
        if self.return_multiscale:
            seg_bev_dict = self.seg_bev_encoder(seg_bev)
            return seg_bev_dict  # {1: [B,C,H,W], 2: [B,C*2,H//2,W//2], 4: [B,C*4,H//4,W//4]}
        else:
            return seg_bev
