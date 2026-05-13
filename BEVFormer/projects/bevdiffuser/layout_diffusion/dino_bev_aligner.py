import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16



class DINOBEVAligner(nn.Module):
    """
    Self-contained BEV aligner for DINOv2 last_tokens using BEVFormer-style reference generation.

    Inputs:
      - last_tokens: (B, V, N, C_dino)
      - patch_hw:    (Hp, Wp) with Hp*Wp == N
      - img_metas:   list of dicts (len=B), each with:
          * 'lidar2img': (V, 4, 4)
          * 'img_shape' : ((H, W, 3),) or similar; we use [0][0]=H, [0][1]=W

    Returns:
      - dino_bev: (B, C_ctx, bev_h, bev_w)
    """
    def __init__(
        self,
        bev_h=50,
        bev_w=50,
        cam_view=6,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        c_dino=768,                 # DINO feature dim
        c_ctx=160,                  # output channels
        channel_mult=[1,2,4],
        post_ln_affine=True,       # recommended True (stability + capacity)    
        return_multiscale=False,
        eps=1e-6,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.c_dino = c_dino
        self.c_feat = c_dino // 2
        self.c_ctx = c_ctx
        self.return_multiscale = return_multiscale
        self.eps = eps
        
        self.channel_reducer = nn.Conv2d(c_dino, self.c_feat, kernel_size=1)

        # Norms are created lazily with correct feature dim
        self.post_ln_affine = post_ln_affine
        # [Improvement 3] Pre-LN enabled for feature normalization before grid sampling
        self.pre_ln = nn.LayerNorm(self.c_feat, elementwise_affine=True)    # TODO: pre_ln 없애기
        self.post_ln = nn.LayerNorm(self.c_feat, elementwise_affine=self.post_ln_affine)

        # Per-view weights
        self._w_view = nn.Parameter(th.zeros(1, cam_view, 1))

        # [Improvement 4] MLP projection instead of simple linear
        # (B,Q,C_feat) -> (B,Q,C_ctx)
        self.proj = nn.Sequential(
            nn.Linear(self.c_feat, self.c_ctx),
            nn.GELU(),
            nn.Linear(self.c_ctx, self.c_ctx),
        )

        # [Improvement 1] Learnable 2D positional embedding for BEV grid
        self.bev_pos_embed = nn.Parameter(th.zeros(1, self.c_ctx, bev_h, bev_w))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        # ---- Multi-scale BEV encoder (64 → 128, 256, 512) ----
        if self.return_multiscale:
            self.dino_bev_encoder = DINOBevEncoder(
                in_channels=self.c_ctx,
                channel_mult=channel_mult,
            )
        
    # ---------- BEVFormer-style reference generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=th.float32):
        if dim == '3d':
            zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = th.stack((xs, ys, zs), -1)                 # (D,H,W,3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (D, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)              # (bs, D, H*W, 3)
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
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)          # (bs, H*W, 1, 2)
            return ref_2d
        else:
            raise ValueError("dim must be '3d' or '2d'")
 
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
        # ❌ 기존: 전역 TF32 플래그 비활성화 → GPU stall + matmul 성능 저하
        # allow_tf32 = th.backends.cuda.matmul.allow_tf32
        # allow_tf32_cudnn  = th.backends.cudnn.allow_tf32
        # th.backends.cuda.matmul.allow_tf32 = False
        # th.backends.cudnn.allow_tf32 = False

        # ✅ 수정: torch.as_tensor로 직접 GPU tensor 생성 → CPU sync 제거
        # (B, V, 4, 4)
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

        # ❌ 기존: allow_tf32 전역 플래그 복원 (비활성화 코드를 제거했으므로 복원도 불필요)
        # th.backends.cuda.matmul.allow_tf32 = allow_tf32
        # th.backends.cudnn.allow_tf32 = allow_tf32_cudnn
        return uv, bev_mask

    def _tokens_to_fmap(self, last_tokens, Hp, Wp):
        B, V, N, C = last_tokens.shape
        fmap = last_tokens.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()  # (B,V,C,Hp,Wp)
        # TODO 수정사항: pre_ln 제거
        if self.pre_ln is not None:
            t = fmap.permute(0,1,3,4,2).reshape(-1, C)  # (B*V*Hp*Wp, C)
            t = self.pre_ln(t)
            fmap = t.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()
        return fmap

    def forward(self, last_tokens, patch_hw, img_metas, dino_geom):
        """
        last_tokens: (B,V,C_dino,Hp,Wp) -- DINO spatial features
        patch_hw:    (Hp,Wp)
        img_metas:   list length B (BEVFormer-like metas)
        dino_geom:   dict with DINO geometry info (scale, padding, H2W2, patch_size)
        returns:     (B, C_ctx, bev_h, bev_w)
        """
        # (0) Channel reduction with optional camera-aware SE modulation
        B, V, C, Hp, Wp = last_tokens.shape
        reduced = self.channel_reducer(last_tokens.reshape(B * V, C, Hp, Wp))
        dino_feat = reduced.reshape(B, V, self.c_feat, Hp, Wp)

        Hp, Wp = patch_hw
        B, V, C_feat, _, _ = dino_feat.shape

        # ✅ [FIX] permute로 spatial-last 배치 후 view → 채널·공간 순서 보존
        dino_feat = dino_feat.permute(0, 1, 3, 4, 2).contiguous().view(B, V, Hp*Wp, C_feat)  # (B,V,N,C_feat)

        # (1) DINO fmap
        fmap = self._tokens_to_fmap(dino_feat, Hp, Wp)  # (B, V, C_feat, Hp, Wp)

        # (2) BEV refs and camera projection → original image pixel coords
        Z_bins = int(round((self.pc_range[5] - self.pc_range[2])))
        ref_3d = self._get_reference_points(self.bev_h, self.bev_w, Z=Z_bins,
                                            num_points_in_pillar=self.num_points_in_pillar,
                                            dim='3d', bs=B, device=fmap.device, dtype=fmap.dtype)
        uv, bev_mask = self.point_sampling(ref_3d, img_metas)  # (V, B, Q, D, 2), (V,B,Q,D)

        # (3) Original pixel → DINO input pixel (no ida transform)
        Q = self.bev_h * self.bev_w
        QD = Q * self.num_points_in_pillar

        # Reshape uv: (V,B,Q,D,2) → (B,V,QD,2)
        uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)

        # Apply DINO preprocessing: original pixel → DINO input pixel
        scale = dino_geom['scale']
        pad_top, pad_left = dino_geom['padding'][0], dino_geom['padding'][1]
        H2, W2 = dino_geom['H2W2'][0], dino_geom['H2W2'][1]

        u = uv_flat[..., 0] * scale + pad_left   # (B, V, QD)
        v = uv_flat[..., 1] * scale + pad_top

        # Validity check (in DINO input pixel space)
        valid_in = (u >= 0) & (u <= (W2 - 1)) & (v >= 0) & (v <= (H2 - 1))
        bev_mask_flat = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD)
        mask_bv = bev_mask_flat & valid_in  # (B, V, QD)

        # Normalise to [-1, 1] for grid_sample (DINO pixel coords directly)
        gx = 2.0 * (u / (W2 - 1.0)) - 1.0   # (B, V, QD)
        gy = 2.0 * (v / (H2 - 1.0)) - 1.0
        grid = th.stack([gx, gy], dim=-1)     # (B, V, QD, 2)

        # (4) bilinear sampling
        fmap_v = fmap.view(B * V, C_feat, Hp, Wp)
        grid_v = grid.view(B * V, QD, 1, 2)

        sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
                                padding_mode='border', align_corners=True)  # (B*V, C, QD, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B*V, QD, C)
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, C_feat)

        # (6) pillar mean + view-weighted mean
        mask = mask_bv.view(B, V, Q, self.num_points_in_pillar).unsqueeze(-1).float()  # (B,V,Q,D,1)
        sampled = sampled * mask
        denom_D = mask.sum(dim=3, keepdim=True).clamp_min(self.eps)        # (B,V,Q,1,1)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D                # (B,V,Q,1,C)
        feat_v = feat_v.squeeze(3)                                         # (B,V,Q,C)

        # view-weighted mean
        w = F.softplus(self._w_view).expand(B, -1, -1)
        w = w.unsqueeze(-1)                                                # (B, V, 1, 1)
        view_valid = (denom_D.squeeze(3) > 0).float()
        num = (feat_v * w).sum(dim=1)                                      # (B,Q,C)
        den = (w * view_valid).sum(dim=1).clamp_min(self.eps)              # (B,Q,1)
        f_bev = num / den                                                  # (B,Q,C_feat)

        # (5) post-norm: aggregation 완료 후 BEV 토큰 단위로 정규화
        f_bev = self.post_ln(f_bev.view(-1, C_feat)).view(B, Q, C_feat)

        # (7) channel reduction (B,Q,C_feat) -> (B,Q,C_ctx)
        bev_feat = self.proj(f_bev)                                       # (B,Q,C_ctx)

        # reshape to (B,C_ctx,H,W)
        dino_bev = bev_feat.permute(0,2,1).contiguous().view(B, self.c_ctx, self.bev_h, self.bev_w)
        
        # [Improvement 1] Add BEV positional encoding
        dino_bev = dino_bev + self.bev_pos_embed
        
        if self.return_multiscale:
            dino_bev_dict = self.dino_bev_encoder(dino_bev)
            return dino_bev_dict
        else:
            return dino_bev


    def forward_single_scale(self, last_tokens, patch_hw, img_metas, dino_geom):
        # (0) Channel reduction
        B, V, C, Hp, Wp = last_tokens.shape
        Hp, Wp = patch_hw
        dino_feat = self.channel_reducer(last_tokens.reshape(B * V, C, Hp, Wp)) \
                        .permute(0, 2, 3, 1).contiguous() \
                        .reshape(B, V, Hp * Wp, self.c_feat)

        # (1) DINO fmap
        fmap = self._tokens_to_fmap(dino_feat, Hp, Wp)

        # (2) BEV refs and camera projection
        Z_bins = int(round((self.pc_range[5] - self.pc_range[2])))
        ref_3d = self._get_reference_points(self.bev_h, self.bev_w, Z=Z_bins,
                                            num_points_in_pillar=self.num_points_in_pillar,
                                            dim='3d', bs=B, device=fmap.device, dtype=fmap.dtype)
        uv, bev_mask = self.point_sampling(ref_3d, img_metas)

        # (3) Original pixel → DINO input pixel
        Q = self.bev_h * self.bev_w
        QD = Q * self.num_points_in_pillar
        uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)

        scale = dino_geom['scale']
        pad_top, pad_left = dino_geom['padding'][0], dino_geom['padding'][1]
        H2, W2 = dino_geom['H2W2'][0], dino_geom['H2W2'][1]

        u = uv_flat[..., 0] * scale + pad_left
        v = uv_flat[..., 1] * scale + pad_top

        valid_in = (u >= 0) & (u <= (W2 - 1)) & (v >= 0) & (v <= (H2 - 1))
        bev_mask_flat = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD)
        mask_bv = bev_mask_flat & valid_in

        gx = 2.0 * (u / (W2 - 1.0)) - 1.0
        gy = 2.0 * (v / (H2 - 1.0)) - 1.0
        grid = th.stack([gx, gy], dim=-1)

        # (4) bilinear sampling
        fmap_v = fmap.view(B * V, self.c_feat, Hp, Wp)
        grid_v = grid.view(B * V, QD, 1, 2)
        sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
                                padding_mode='border', align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, self.c_feat)

        # (5) pillar mean + view-weighted mean
        mask = mask_bv.view(B, V, Q, self.num_points_in_pillar).unsqueeze(-1).float()
        sampled = sampled * mask
        denom_D = mask.sum(dim=3, keepdim=True).clamp_min(self.eps)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D
        feat_v = feat_v.squeeze(3)

        w = F.softplus(self._w_view).expand(B, -1, -1)
        w = w.unsqueeze(-1)
        view_valid = (denom_D.squeeze(3) > 0).float()
        num = (feat_v * w).sum(dim=1)
        den = (w * view_valid).sum(dim=1).clamp_min(self.eps)
        f_bev = num / den

        # (6) post-norm
        f_bev = self.post_ln(f_bev.view(-1, self.c_feat)).view(B, Q, self.c_feat)

        # (7) channel reduction
        bev_feat = self.proj(f_bev)

        # reshape to (B, C_ctx, H, W)
        dino_bev = bev_feat.permute(0,2,1).contiguous().view(B, self.c_ctx, self.bev_h, self.bev_w)
        dino_bev = dino_bev + self.bev_pos_embed
        return dino_bev


class DINOBevEncoder(nn.Module):
    """
    Multi-scale BEV encoder with channel-preserving depthwise stride-2 convs.

    채널을 고정한 채로 resolution만 줄입니다:
        ds=1 -> (B, C, bevH,    bevW)     e.g. (B, 256, 50, 50)  pass-through
        ds=2 -> (B, C, bevH//2, bevW//2)  e.g. (B, 256, 25, 25)
        ds=4 -> (B, C, bevH//4, bevW//4)  e.g. (B, 256, 12, 12)

    Depthwise conv(groups=C)를 사용하므로 채널 간 mixing 없이
    공간적 aggregation만 수행 → 원본 BEV 표현을 보존하면서 resolution만 압축.
    avg_pool 대비 학습 가능한 spatial aggregation으로 경계 정보 손실을 줄임.

    # ── 구버전 (채널 확장 방식) ──────────────────────────────────────────────
    # ds=1 -> (B, C*1, bevH,    bevW)
    # ds=2 -> (B, C*2, bevH//2, bevW//2)
    # ds=4 -> (B, C*4, bevH//4, bevW//4)
    # ─────────────────────────────────────────────────────────────────────────
    """
    def __init__(self, in_channels, channel_mult=(1, 2, 4)):
        super().__init__()
        C = in_channels

        # ds=1: pass-through (no op)
        # ds=2: depthwise stride-2 conv — 채널 고정, 50->25
        # self.down1 = nn.Sequential(
        #     nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C),
        #     nn.SiLU(),
        # )
        # # ds=4: depthwise stride-2 conv — 채널 고정, 25->12 (padding=0: floor((25-3)/2)+1=12)
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(C, C, kernel_size=3, stride=2, padding=0, groups=C),
        #     nn.SiLU(),
        # )

        # ── 구버전 (채널 확장) ─────────────────────────────────────────────
        c1 = in_channels * channel_mult[0]
        c2 = in_channels * channel_mult[1]
        c4 = in_channels * channel_mult[2]
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c4, 3, stride=2, padding=0),
            nn.SiLU(),
        )
        # ──────────────────────────────────────────────────────────────────

    def forward(self, bev_ctx):
        # s1 = bev_ctx        # (B, C, 50, 50)  pass-through
        # s2 = self.down1(s1) # (B, C, 25, 25)
        # s4 = self.down2(s2) # (B, C, 12, 12)

        # ── 구버전 ─────────────────────────────────────────────────────────
        s1 = bev_ctx                # (B, C*1, bevH,    bevW)
        s2 = self.down1(s1)         # (B, C*2, bevH//2, bevW//2)
        s4 = self.down2(s2)         # (B, C*4, bevH//4, bevW//4)
        # ──────────────────────────────────────────────────────────────────

        return {
            1: s1,
            2: s2,
            4: s4,
        }




class DINODeformAligner(nn.Module):
    """
    BEVFormer-style deformable attention aligner for DINOv2 2D features.

    Unlike DINOBevAligner (which aggregates features to BEV), this module:
      1. Applies camera-aware feature modulation (CamAwareDINO)
      2. Pre-computes geometric reference UV anchors for BEV queries at 3 scales

    Inputs:
      - last_tokens: (B, V, C_dino, Hp, Wp)
      - img_metas:   list of dicts with 'lidar2img' (V, 4, 4)
      - dino_geom:   dict with scale, padding, H2W2

    Returns:
      {
        'fmap':     (B*V, C_feat, Hp, Wp)          DINOv2 feature map for grid_sample
        'ref_uvs':  {1: (B,V,Q1,D,2), 2:..., 4:...} anchor UV coords in [-1,1]
        'bev_mask': {1: (B,V,Q1,D),   2:..., 4:...} validity masks
      }
    """

    def __init__(
        self,
        bev_h=64,
        bev_w=64,
        cam_view=6,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        c_dino=768,
        c_ctx=256,
        eps=1e-6,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.c_dino = c_dino
        self.c_feat = c_ctx
        self.eps = eps

        self.channel_reducer = nn.Conv2d(c_dino, self.c_feat, kernel_size=1, bias=True)
        self.pre_ln = nn.LayerNorm(self.c_feat, elementwise_affine=True)

    # ---------- BEVFormer-style reference generation (same as DINOBevAligner) ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d',
                               bs=1, device='cuda', dtype=th.float32):
        if dim == '3d':
            zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                             device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = th.linspace(0.5, W - 0.5, W, dtype=dtype,
                             device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = th.linspace(0.5, H - 0.5, H, dtype=dtype,
                             device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = th.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)   # (D, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)                          # (bs, D, H*W, 3)
            return ref_3d
        else:
            raise ValueError("DINODeformAligner only supports dim='3d'")

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
        # ❌ 기존: 전역 TF32 플래그 비활성화 → GPU stall + matmul 성능 저하
        # allow_tf32 = th.backends.cuda.matmul.allow_tf32
        # allow_tf32_cudnn = th.backends.cudnn.allow_tf32
        # th.backends.cuda.matmul.allow_tf32 = False
        # th.backends.cudnn.allow_tf32 = False

        # ✅ 수정: torch.as_tensor로 직접 GPU tensor 생성 → CPU sync 제거
        lidar2img = th.stack([
            th.as_tensor(np.array(m['lidar2img']), dtype=th.float32, device=reference_points.device)
            for m in img_metas
        ])  # (B, V, 4, 4)

        pc_range = self.pc_range
        ref = reference_points.clone()
        ref[..., 0:1] = ref[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref = th.cat((ref, th.ones_like(ref[..., :1])), -1)   # (bs, D, Q, 4)

        ref = ref.permute(1, 0, 2, 3)                          # (D, B, Q, 4)
        D, B, Q = ref.size()[:3]
        num_cam = lidar2img.size(1)

        ref = ref.view(D, B, 1, Q, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, Q, 1, 1)

        cam = th.matmul(lidar2img.to(th.float32), ref.to(th.float32)).squeeze(-1)  # (D,B,V,Q,4)
        eps = 1e-5
        depth = cam[..., 2:3]
        bev_mask = (depth > eps)                                                    # (D,B,V,Q,1)

        uv = cam[..., 0:2] / th.maximum(depth, th.ones_like(depth) * eps)          # (D,B,V,Q,2)

        uv      = uv.permute(2, 1, 3, 0, 4).contiguous()           # (V,B,Q,D,2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()        # (V,B,Q,D)

        # ❌ 기존: allow_tf32 전역 플래그 복원
        # th.backends.cuda.matmul.allow_tf32 = allow_tf32
        # th.backends.cudnn.allow_tf32 = allow_tf32_cudnn
        return uv, bev_mask

    def forward(self, last_tokens, img_metas, dino_geom):
        """
        last_tokens: (B, V, C_dino, Hp, Wp)
        """
        # (0) Channel reduction
        B, V, _, Hp, Wp = last_tokens.shape
        dino_reduced = self.channel_reducer(last_tokens.reshape(B * V, self.c_dino, Hp, Wp))
        dino_feat = dino_reduced.reshape(B, V, self.c_feat, Hp, Wp)
        B, V, C_feat, Hp, Wp = dino_feat.shape

        # (1) Pre-LN
        t = dino_feat.permute(0, 1, 3, 4, 2).reshape(-1, C_feat)
        t = self.pre_ln(t)
        dino_feat = t.reshape(B, V, Hp, Wp, C_feat).permute(0, 1, 4, 2, 3).contiguous()

        # (2) Flatten to (B*V, C_feat, Hp, Wp) for grid_sample
        dino_fmap = dino_feat.reshape(B * V, C_feat, Hp, Wp)   # batch-major: [b0v0, b0v1, ...]

        # (3) Geometry: original pixel → DINO input pixel (no ida transform)
        scale_g  = dino_geom['scale']
        pad_top, pad_left = dino_geom['padding'][0], dino_geom['padding'][1]
        H2, W2   = dino_geom['H2W2'][0], dino_geom['H2W2'][1]
        Z_bins   = int(round((self.pc_range[5] - self.pc_range[2])))
        device   = dino_fmap.device
        dtype    = dino_fmap.dtype

        ref_uvs_dict  = {}
        bev_mask_dict = {}

        for scale in [1, 2, 4]:
            h_bev = self.bev_h // scale
            w_bev = self.bev_w // scale
            Q = h_bev * w_bev
            D = self.num_points_in_pillar

            # (a) 3D reference points: (B, D, Q, 3)
            ref_3d = self._get_reference_points(h_bev, w_bev, Z=Z_bins,
                                                num_points_in_pillar=D,
                                                dim='3d', bs=B,
                                                device=device, dtype=dtype)

            # (b) Project to camera pixel coords
            uv_raw, bev_mask_raw = self.point_sampling(ref_3d, img_metas)

            # (c) Permute to (B, V, Q, D, 2/bool)
            uv_bvqd    = uv_raw.permute(1, 0, 2, 3, 4).contiguous()      # (B, V, Q, D, 2)
            mask_bvqd  = bev_mask_raw.permute(1, 0, 2, 3).contiguous()   # (B, V, Q, D)

            # (d) DINO geom transform: original pixel → DINO input pixel → normalize [-1,1]
            QD = Q * D
            uv_flat = uv_bvqd.reshape(B, V, QD, 2)

            u_px = uv_flat[..., 0] * scale_g + pad_left   # (B, V, QD)
            v_px = uv_flat[..., 1] * scale_g + pad_top

            valid_in   = (u_px >= 0) & (u_px <= (W2 - 1)) & (v_px >= 0) & (v_px <= (H2 - 1))
            mask_flat  = mask_bvqd.reshape(B, V, QD) & valid_in          # (B, V, QD)

            gx = 2.0 * (u_px / (W2 - 1.0)) - 1.0   # (B, V, QD)
            gy = 2.0 * (v_px / (H2 - 1.0)) - 1.0
            grid = th.stack([gx, gy], dim=-1)         # (B, V, QD, 2)

            # Reshape back to (B, V, Q, D, 2) and (B, V, Q, D)
            ref_uvs_dict[scale]  = grid.reshape(B, V, Q, D, 2)
            bev_mask_dict[scale] = mask_flat.reshape(B, V, Q, D)

        return {
            'fmap':     dino_fmap,
            'ref_uvs':  ref_uvs_dict,
            'bev_mask': bev_mask_dict,
        }
