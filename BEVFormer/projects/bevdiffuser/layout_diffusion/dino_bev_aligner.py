import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16

class DINOBEVAligner(nn.Module):
    """
    Inputs:
      - last_tokens: (B, V, N, C_dino)
      - patch_hw:    (Hp, Wp) with Hp*Wp == N
      - img_metas:   list of dicts (len=B), each with:
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
        post_ln_affine=True,       # recommended True (stability + capacity)
        eps=1e-6,
        final_dim=(480, 800),       # augmented image size (H, W) — DA3 depth가 정의된 FOV
        use_bev_pos_embed=False,
        # ---- Depth Consistency (FB-BEV) params ----
        depth_consistency_mode=None,        # 'gaussian' | 'bin_linear' | None
        depth_consistency_sigma=2.0,        # Gaussian mode: 허용 오차 σ (meters)
        d_bound=(2.0, 58.0, 0.5),           # bin_linear mode: (start, end, step) in meters
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.c_dino = c_dino
        self.c_feat = c_dino
        self.c_ctx = c_ctx
        self.eps = eps
        self.final_dim = final_dim
        self.use_bev_pos_embed = use_bev_pos_embed
        # ---- Depth Consistency config ----
        self.depth_consistency_mode = depth_consistency_mode
        self.depth_consistency_sigma = depth_consistency_sigma
        self.d_bound = d_bound
        self.depth_min = d_bound[0]
        self.depth_max = d_bound[1]
        self.depth_step = d_bound[2]
        self.num_depth_bins = int(round((d_bound[1] - d_bound[0]) / d_bound[2]))

        # Norms are created lazily with correct feature dim
        self.post_ln_affine = post_ln_affine
        self.post_ln = nn.LayerNorm(self.c_feat, elementwise_affine=self.post_ln_affine)

        # Per-view weights
        self._w_view = nn.Parameter(th.zeros(1, cam_view, 1))

        # (B,Q,C_dino) -> (B,Q,C_ctx)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.c_feat, self.c_ctx),
        #     nn.GELU(),
        #     nn.Linear(self.c_ctx, self.c_ctx),
        # )
        self.proj = nn.Linear(self.c_dino, self.c_ctx, bias=True)

        if self.use_bev_pos_embed:
            self.bev_pos_embed = nn.Parameter(th.zeros(1, self.c_ctx, bev_h, bev_w))
            nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        # ---- Multi-scale BEV encoder (64 → 128, 256, 512) ----
        # self.dino_bev_encoder = DINOBevEncoder(
        #     in_channels=self.c_ctx,
        #     channel_mult=channel_mult,
        # )
        
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
        depth_out = depth.squeeze(-1).permute(2, 1, 3, 0).contiguous()  # (V, B, Q, D)

        # ❌ 기존: allow_tf32 전역 플래그 복원 (비활성화 코드를 제거했으므로 복원도 불필요)
        # th.backends.cuda.matmul.allow_tf32 = allow_tf32
        # th.backends.cudnn.allow_tf32 = allow_tf32_cudnn
        return uv, bev_mask, depth_out

    # ---------- FB-BEV Depth Consistency (w_c) ----------
    def _compute_depth_consistency(self, proj_depth, da3_depth):
        if self.depth_consistency_mode == 'gaussian':
            # Gaussian 커널: 깊이 차이가 작을수록 w_c → 1, 클수록 → 0
            # 수식: w_c = exp(-(d_proj - d_da3)^2 / (2 * σ^2))
            diff_sq = (proj_depth - da3_depth).pow(2)
            w_c = th.exp(-diff_sq / (2.0 * self.depth_consistency_sigma ** 2))

        elif self.depth_consistency_mode == 'bin_linear':
            bin_width = self.depth_step  # d_bound[2]

            proj_idx = ((proj_depth - self.depth_min) / bin_width).clamp(0, self.num_depth_bins - 1 - 1e-3)
            proj_lo = proj_idx.floor().long()                                   # [B, V, Q, D]
            proj_hi = (proj_lo + 1).clamp(max=self.num_depth_bins - 1)          # [B, V, Q, D]
            proj_frac = proj_idx - proj_lo.float()                              # [B, V, Q, D]

            # da3_depth 이산화
            da3_idx = ((da3_depth - self.depth_min) / bin_width).clamp(0, self.num_depth_bins - 1 - 1e-3)
            da3_lo = da3_idx.floor().long()                                     # [B, V, Q, D]
            da3_hi = (da3_lo + 1).clamp(max=self.num_depth_bins - 1)            # [B, V, Q, D]
            da3_frac = da3_idx - da3_lo.float()                                 # [B, V, Q, D]
            
            w_c = (
                (proj_lo == da3_lo).float() * (1 - proj_frac) * (1 - da3_frac) +  # lo-lo
                (proj_lo == da3_hi).float() * (1 - proj_frac) * da3_frac       +  # lo-hi
                (proj_hi == da3_lo).float() * proj_frac       * (1 - da3_frac) +  # hi-lo
                (proj_hi == da3_hi).float() * proj_frac       * da3_frac           # hi-hi
            )
        else:
            return th.ones_like(proj_depth)

        w_c = w_c * (da3_depth > 0.5).float()
        return w_c

    def _tokens_to_fmap(self, last_tokens, Hp, Wp):
        B, V, N, C = last_tokens.shape
        fmap = last_tokens.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()  # (B,V,C,Hp,Wp)
        # pre_ln 제거됨 (정규화는 aggregation 후 post_ln에서 수행)
        return fmap

    def _get_image_hw(self, dino_geom):
        """Return the BEVFormer-padded image canvas size before DINO patch padding."""
        if isinstance(dino_geom, dict) and 'input_hw' in dino_geom:
            return int(dino_geom['input_hw'][0]), int(dino_geom['input_hw'][1])
        return int(self.final_dim[0]), int(self.final_dim[1])

    def forward(self, last_tokens, patch_hw, img_metas, dino_geom, depth_maps=None):
        """
        last_tokens: DINO spatial features from GetDINOV2Feat, either
                       (B, T, V, C_dino, Hp, Wp)  -- T key frames, or
                       (B, V, C_dino, Hp, Wp)     -- single frame.
        patch_hw:    (Hp,Wp)
        img_metas:   list length B (BEVFormer-like metas)
        dino_geom:   dict with DINO geometry info (scale, padding, H2W2, patch_size)
        depth_maps:  (B, V, dH, dW) DA3 예측 깊이 (original/augmented FOV). None이면 가중치 미적용
        returns:     (B, C_ctx, bev_h, bev_w)
        """

        if last_tokens.dim() == 6:           # (B, T, V, C_dino, Hp, Wp)
            last_tokens = last_tokens[:, -1]  # current frame → (B, V, C_dino, Hp, Wp)
        assert last_tokens.dim() == 5, f'expected (B,V,C,Hp,Wp), got {tuple(last_tokens.shape)}'
        B, V, C, Hp, Wp = last_tokens.shape
        assert C == self.c_dino, f'expected C={self.c_dino}, got {C}'
        dino_feat = last_tokens

        feat_Hp, feat_Wp = Hp, Wp
        Hp, Wp = patch_hw
        assert (Hp, Wp) == (feat_Hp, feat_Wp), (
            f'patch_hw {patch_hw} mismatches feature map {(feat_Hp, feat_Wp)}'
        )
        B, V, C_feat, _, _ = dino_feat.shape

        # spatial-last layout preserves channel/spatial order before flattening.
        dino_feat = dino_feat.permute(0, 1, 3, 4, 2).contiguous().view(B, V, Hp*Wp, C_feat)  # (B,V,N,C_feat)

        # (1) DINO fmap
        fmap = self._tokens_to_fmap(dino_feat, Hp, Wp)  # (B, V, C_feat, Hp, Wp)

        # (2) BEV refs and camera projection → original image pixel coords
        Z_bins = int(round((self.pc_range[5] - self.pc_range[2])))
        ref_3d = self._get_reference_points(self.bev_h, self.bev_w, Z=Z_bins,
                                            num_points_in_pillar=self.num_points_in_pillar,
                                            dim='3d', bs=B, device=fmap.device, dtype=fmap.dtype)
        uv, bev_mask, proj_depth = self.point_sampling(ref_3d, img_metas)  # (V,B,Q,D,2), (V,B,Q,D), (V,B,Q,D)

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

        imgH, imgW = self._get_image_hw(dino_geom)
        valid_in = (uv_flat[..., 0] >= 0) & (uv_flat[..., 0] <= (imgW - 1)) & \
                   (uv_flat[..., 1] >= 0) & (uv_flat[..., 1] <= (imgH - 1))
        valid_dino = (u >= 0) & (u <= (W2 - 1)) & (v >= 0) & (v <= (H2 - 1))
        bev_mask_flat = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD)
        mask_bv = bev_mask_flat & valid_in & valid_dino  # (B, V, QD)
        
        gx = 2.0 * (u + 0.5) / W2 - 1.0   # (B, V, QD)
        gy = 2.0 * (v + 0.5) / H2 - 1.0
        grid = th.stack([gx, gy], dim=-1)     # (B, V, QD, 2)

        # (4) bilinear sampling
        fmap_v = fmap.view(B * V, C_feat, Hp, Wp)
        grid_v = grid.view(B * V, QD, 1, 2)

        sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
                                padding_mode='border', align_corners=False)  # (B*V, C, QD, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B*V, QD, C)
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, C_feat)

        # (6) Depth Consistency Weighting + Masking
        mask_bvqd = mask_bv.view(B, V, Q, self.num_points_in_pillar)  # (B,V,Q,D) bool

        if depth_maps is not None and self.depth_consistency_mode is not None:
            dH, dW = depth_maps.shape[-2:]                                      # e.g. (448, 798)

            # (6a) DA3 depth -> BEVFormer-padded image canvas resize.
            da3 = F.interpolate(
                depth_maps.reshape(B * V, 1, dH, dW).to(sampled.dtype),
                size=(imgH, imgW), mode='bilinear', align_corners=True
            )                                                                  # [B*V, 1, imgH, imgW]

            # (6b) DINO transform 이전의 original/augmented (u,v) 좌표로 DA3 grid 구성
            u_orig = uv_flat[..., 0]  # (B, V, QD)
            v_orig = uv_flat[..., 1]
            gx_da3 = 2.0 * (u_orig / (imgW - 1.0)) - 1.0
            gy_da3 = 2.0 * (v_orig / (imgH - 1.0)) - 1.0
            grid_v_da3 = th.stack([gx_da3, gy_da3], dim=-1).view(B * V, QD, 1, 2)

            # (6c) 투영된 (u,v) 좌표에서 DA3 depth 샘플링
            da3_sampled = F.grid_sample(
                da3, grid_v_da3, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )                                                                  # [B*V, 1, QD, 1]
            da3_sampled = da3_sampled.squeeze(1).squeeze(-1)                    # [B*V, QD]
            da3_sampled = da3_sampled.view(B, V, Q, self.num_points_in_pillar)  # [B, V, Q, D]

            # (6d) proj_depth 차원 재배치: (V,B,Q,D) → (B,V,Q,D)
            proj_depth_bvqd = proj_depth.permute(1, 0, 2, 3).contiguous()      # [B, V, Q, D]

            # (6e) Depth consistency 가중치 계산 후 마스크와 동시 적용
            w_c = self._compute_depth_consistency(proj_depth_bvqd, da3_sampled) # [B, V, Q, D]
            weight = mask_bvqd.float() * w_c                                    # [B, V, Q, D]
        else:
            # depth_maps 미제공 시 기존 이진 마스크만 적용 (bit-identical fallback)
            weight = mask_bvqd.float()

        # (7) pillar mean + view-weighted mean
        mask = weight.unsqueeze(-1)                                        # (B,V,Q,D,1)
        sampled = sampled * mask
        denom_D_raw = mask.sum(dim=3, keepdim=True)                        # (B,V,Q,1,1)
        denom_D = denom_D_raw.clamp_min(self.eps)                          # (B,V,Q,1,1)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D                # (B,V,Q,1,C)
        feat_v = feat_v.squeeze(3)                                         # (B,V,Q,C)

        # view-weighted mean
        w = F.softplus(self._w_view).expand(B, -1, -1)
        w = w.unsqueeze(-1)                                                # (B, V, 1, 1)
        view_valid = (denom_D_raw.squeeze(3) > 0).float()
        num = (feat_v * w).sum(dim=1)                                      # (B,Q,C)
        den_raw = (w * view_valid).sum(dim=1)                               # (B,Q,1)
        den = den_raw.clamp_min(self.eps)                                   # (B,Q,1)
        f_bev = num / den                                                  # (B,Q,C_feat)
        bev_valid = (den_raw > 0).to(f_bev.dtype)

        # (5) post-norm: aggregation 완료 후 BEV 토큰 단위로 정규화
        f_bev = self.post_ln(f_bev.view(-1, C_feat)).view(B, Q, C_feat)

        # (7) projection (B,Q,C_dino) -> (B,Q,C_ctx)
        bev_feat = self.proj(f_bev)                                       # (B,Q,C_ctx)

        # reshape to (B,C_ctx,H,W)
        dino_bev = bev_feat.permute(0,2,1).contiguous().view(B, self.c_ctx, self.bev_h, self.bev_w)

        if self.use_bev_pos_embed:
            dino_bev = dino_bev + self.bev_pos_embed
        dino_bev = dino_bev * bev_valid.permute(0, 2, 1).contiguous().view(B, 1, self.bev_h, self.bev_w)

        # dino_bev_dict = self.dino_bev_encoder(dino_bev)
        # return dino_bev_dict
        
        return dino_bev



class DINOBevEncoder(nn.Module):
    """
    채널을 고정한 채로 resolution만 줄입니다:
        ds=1 -> (B, C, bevH,    bevW)     e.g. (B, 256, 50, 50)  pass-through
        ds=2 -> (B, C, bevH//2, bevW//2)  e.g. (B, 256, 25, 25)
        ds=4 -> (B, C, bevH//4, bevW//4)  e.g. (B, 256, 12, 12)
    """
    def __init__(self, in_channels, channel_mult=(1, 2, 4)):
        super().__init__()
        C = in_channels

        # ds=1: pass-through (no op)
        # ds=2: depthwise stride-2 conv — 채널 고정, 50->25
        self.down1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C),
            nn.SiLU(),
        )
        # ds=4: depthwise stride-2 conv — 채널 고정, 25->12 (padding=0: floor((25-3)/2)+1=12)
        self.down2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=0, groups=C),
            nn.SiLU(),
        )

        # # ── 구버전 (채널 확장) ─────────────────────────────────────────────
        # c1 = in_channels * channel_mult[0]
        # c2 = in_channels * channel_mult[1]
        # c4 = in_channels * channel_mult[2]
        # self.down1 = nn.Sequential(
        #     nn.Conv2d(c1, c2, 3, stride=2, padding=1),
        #     nn.SiLU(),
        # )
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(c2, c4, 3, stride=2, padding=0),
        #     nn.SiLU(),
        # )
        # # ──────────────────────────────────────────────────────────────────

    def forward(self, bev_ctx):
        
        s1 = bev_ctx                # (B, C*1, bevH,    bevW)
        s2 = self.down1(s1)         # (B, C*2, bevH//2, bevW//2)
        s4 = self.down2(s2)         # (B, C*4, bevH//4, bevW//4)

        return {
            1: s1,
            2: s2,
            4: s4,
        }
