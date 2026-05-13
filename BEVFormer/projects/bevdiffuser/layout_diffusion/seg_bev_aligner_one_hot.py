import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

class SegEmbedEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.downsample_factor = 2

    def forward(self, seg_id):
        B, V, H, W = seg_id.shape
        seg_id = seg_id.view(B * V, H, W)

        seg_id = seg_id.clamp(min=-1, max=self.num_classes - 1)
        seg_id = torch.where(seg_id == -1, torch.zeros_like(seg_id), seg_id)

        # Nearest downsampling (이산 ID에 수학적으로 유효하며, One-hot 연산 이전에 실행하여 메모리 최적화)
        seg_id_ds = F.interpolate(
            seg_id.unsqueeze(1).float(),
            scale_factor=0.5, mode='nearest'
        ).squeeze(1).long()                           # [B*V, H//2, W//2]

        seg_oh = F.one_hot(seg_id_ds, num_classes=self.num_classes + 1).float()        # [B*V, H//2, W//2, num_classes+1]
        seg_oh = seg_oh.permute(0, 3, 1, 2).contiguous()  # [B*V, num_classes+1, H//2, W//2]
        return seg_oh


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

        # ds=1: Spatial processing (해상도/채널 유지)
        self.down0 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        # ds=2: Strided Conv (학습 기반 다운샘플 + 채널 확장 c1→c2)
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        # ds=4: Strided Conv (학습 기반 다운샘플 + 채널 확장 c2→c4)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c4, kernel_size=3, stride=2),  # c1→c2 아님, down1 출력(c2)을 받음
            nn.SiLU(),
        )

    def forward(self, bev_ctx):
        """
        Args:
            bev_ctx: (B, C, bevH, bevW)
        Returns:
            dict {1: (B, C1, H1, W1), 2: (B, C2, H2, W2), 4: (B, C4, H4, W4)}
        """
        s1 = self.down0(bev_ctx)    # (B, 256, 50, 50)
        s2 = self.down1(s1)         # (B, 512, 25, 25)
        s4 = self.down2(s2)         # (B, 1024, 12, 12)

        return {
            1: s1,
            2: s2,
            4: s4,
        }


class SegBEVAligner(nn.Module):
    """
    BEVFormer-style IPM (Inverse Perspective Mapping) for multi-view
    segmentation ID map -> BEV feature. (Soft Histogram Embedding 방식)

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
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        num_classes=16,
        emb_channels=256,           # 최종 BEV 임베딩 채널 수 (UNet의 FDN 모델 채널에 맞춤)
        final_dim=(252, 700),       # augmented image size (H, W)
        channel_mult=[1,2,4],
        eps=1e-6,
        v_min_frac=0.0,             # 이미지 상단 배제 비율 [0, 1): e.g. 0.3 → 상위 30%(하늘) 배제
        vote_weight_mode=None,      # None | 'density' | 'depth_density'
        density_tau=10.0,           # depth-softmax 온도 (미터), depth_density 모드 전용
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.num_classes = num_classes
        self.emb_channels = emb_channels
        self.final_dim = final_dim
        self.eps = eps
        self.v_min_frac = v_min_frac
        self.vote_weight_mode = vote_weight_mode
        self.density_tau = density_tau
        
        # ---- One-hot Segmentation Encoder (설계 간소화, 파라미터 없음) ----
        self.seg_encoder = SegEmbedEncoder(num_classes=num_classes)

        # ---- Projection from Probability to Embedding ----
        self.prob_to_emb = nn.Sequential(
            nn.Conv2d(num_classes + 1, self.emb_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.emb_channels, self.emb_channels, kernel_size=1)
        )

        # ---- Learnable 2D BEV positional embedding ----
        self.bev_pos_embed = nn.Parameter(th.zeros(1, self.emb_channels, bev_h, bev_w))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        # ---- Multi-scale BEV encoder ----
        self.seg_bev_encoder = SegBEVEncoder(
            in_channels=self.emb_channels,
            channel_mult=channel_mult,
        )

    # ---------- Depth-Ordered Pixel Density Normalization (DOPDN) ----------
    def _compute_density_weights(self, uv, depth_vbqd, mask_bvqd):
        """
        Pixel-density normalization with optional depth ordering.

        핵심 원리: 각 카메라의 각 픽셀이 투표권 = 1을 보유하고,
        해당 픽셀에 투영된 모든 BEV 셀에 depth-ordered softmax로 분배한다.

        기하학적 근거:
          Backward projection에서 동일 카메라 ray 상의 여러 BEV 셀이 같은 픽셀에
          투영되는 것이 smearing의 근본 원인이다. 픽셀 단위로 총 투표권을 1로
          고정하면, 같은 픽셀을 공유하는 셀 수가 많을수록 개별 셀의 투표 영향력이
          자동으로 감소한다. depth 기반 softmax로 가까운 셀에 우선권을 부여하여
          occlusion 순서도 반영한다.

          예) 픽셀 p에 3개 셀 투영 (10m, 30m, 50m), τ=10:
              w = softmax([-1, -3, -5]) ≈ [0.91, 0.08, 0.01]
              → 가까운 셀이 91% 투표권, 먼 셀은 1%로 smearing 억제

        mode별 동작:
          - 'density':       raw_w = 1 (uniform) → w = 1/|S(v,p)| 균등 분배
          - 'depth_density': raw_w = exp(-d/τ)   → depth-ordered softmax 분배

        Args:
            uv:          (V, B, Q, D, 2) 원본 해상도 픽셀 좌표 (point_sampling 출력)
            depth_vbqd:  (V, B, Q, D)   카메라 광학축 Z 거리 (미터)
            mask_bvqd:   (B, V, Q, D)   유효성 불리언 마스크
        Returns:
            weights:     (B, V, Q, D) 정규화된 가중치; per-pixel sum ≈ 1
        """
        B, V, Q, D = mask_bvqd.shape
        device = mask_bvqd.device
        ds = self.seg_encoder.downsample_factor           # 2
        imgH, imgW = self.final_dim
        fH, fW = imgH // ds, imgW // ds
        num_px = fH * fW
        BV = B * V

        # (1) UV → downsampled feature-map 픽셀 인덱스
        #     grid_sample(mode='nearest')가 실제 읽는 해상도와 일치시켜야
        #     "같은 관측을 공유하는 셀"을 정확히 그룹화할 수 있음
        uv_bvqd = uv.permute(1, 0, 2, 3, 4)              # (B, V, Q, D, 2)
        u_feat = (uv_bvqd[..., 0] / ds).long().clamp(0, fW - 1)
        v_feat = (uv_bvqd[..., 1] / ds).long().clamp(0, fH - 1)
        px_idx = v_feat * fW + u_feat                      # (B, V, Q, D)

        # (2) Flatten + BV offset → 전역 고유 인덱스
        #     (batch, view) 쌍을 단일 flat index에 인코딩하여 scatter_add 한 번으로 처리
        px_flat = px_idx.reshape(BV, Q * D)
        bv_offset = th.arange(BV, device=device).unsqueeze(1) * num_px
        px_global = (px_flat + bv_offset).reshape(BV * Q * D)

        # (3) Raw weight: depth-ordered softmax 분자 또는 uniform
        depth_bvqd = depth_vbqd.permute(1, 0, 2, 3).contiguous()  # (B, V, Q, D)
        mask_f = mask_bvqd.float()

        if self.vote_weight_mode == 'depth_density':
            # exp(-d/τ): 가까운 관측에 지수적으로 높은 분자값
            # mask_f 곱셈으로 무효 투영은 분자=0 → 분모에 기여하지 않음
            raw_w = th.exp(-depth_bvqd / self.density_tau) * mask_f
        else:  # 'density': 균등 밀도 정규화
            raw_w = mask_f

        raw_flat = raw_w.reshape(BV * Q * D)

        # (4) Per-pixel 분모: scatter_add로 O(N) 집계
        #     각 픽셀에 투영된 모든 셀의 raw_w 합 = softmax 분모
        denom_buf = th.zeros(BV * num_px, dtype=raw_flat.dtype, device=device)
        denom_buf.scatter_add_(0, px_global, raw_flat)

        # (5) 각 투영점의 분모 lookup → 정규화
        per_denom = denom_buf.gather(0, px_global).reshape(B, V, Q, D)
        weights = raw_w / per_denom.clamp_min(self.eps)
        return weights * mask_f

    # ---------- BEVFormer-style 3D reference point generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4,
                              dim='3d', bs=1, device='cuda', dtype=th.float32,):
        if dim == '3d':
            zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar,
                                 dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = th.linspace(0.5, W - 0.5, W,
                             dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = th.linspace(0.5, H - 0.5, H,
                             dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
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

    # ---------- 3D -> 2D camera projection ----------
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
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

        # depth를 보존하여 이후 Distance-aware Weighted Voting에 활용
        # depth[D,B,N,Q,1] → (V,B,Q,D): 카메라 광학축 Z좌표 = occlusion 순서 결정 핵심 정보
        depth_out = depth.squeeze(-1).permute(2, 1, 3, 0).contiguous()  # (V, B, Q, D)

        return uv, bev_mask, depth_out

    def forward(self, seg_id, img_metas):
        """
        Args:
            seg_id:    [B, V, H, W] - multi-view segmentation class-id map
            img_metas: list[dict] (len=B), each with 'lidar2img': (V,4,4)
        Returns:
            dict {1: [B,C,H,W], 2: [B,C*2,H//2,W//2], 4: [B,C*4,H//4,W//4]}
        """
        assert seg_id.dim() == 4, f'seg_id is {seg_id.shape}'
        B, V, H, W = seg_id.shape
        device = seg_id.device

        # (1) Embed segmentation IDs -> one-hot feature maps
        # seg_emb shape: [B*V, num_classes+1, fH, fW]
        seg_emb = self.seg_encoder(seg_id)
        C = self.num_classes + 1

        # (2) Generate BEV 3D reference points
        Z_bins = int(round(self.pc_range[5] - self.pc_range[2]))
        ref_3d = self._get_reference_points(
            self.bev_h, self.bev_w, Z=Z_bins,
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d', bs=B, device=device, dtype=seg_emb.dtype,
        )

        # (3) Project 3D refs to 2D image coords (depth를 보존하여 가중 투표에 활용)
        uv, bev_mask, proj_depth = self.point_sampling(ref_3d, img_metas)  # (V,B,Q,D,2), (V,B,Q,D), (V,B,Q,D)

        # (4) Original pixel → normalised coords for grid_sample
        Q = self.bev_h * self.bev_w
        QD = Q * self.num_points_in_pillar
        imgH, imgW = self.final_dim  # original image size

        # Reshape uv: (V,B,Q,D,2) → (B,V,QD,2)
        uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)

        u = uv_flat[..., 0]  # (B, V, QD)
        v = uv_flat[..., 1]  # (B, V, QD)

        # Check in-bounds validity (in original image space)
        # v_sky_thresh: 이미지 상단 v_min_frac 비율을 배제하여 하늘 영역 샘플링 방지
        v_sky_thresh = self.v_min_frac * imgH
        valid_in = (u >= 0) & (u <= (imgW - 1)) & (v >= v_sky_thresh) & (v <= (imgH - 1))
        bev_mask_flat = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD)
        mask_bv = bev_mask_flat & valid_in  # (B, V, QD)

        # Normalise to [-1, 1] for grid_sample
        gx = 2.0 * (u / (imgW - 1.0)) - 1.0  # (B, V, QD)
        gy = 2.0 * (v / (imgH - 1.0)) - 1.0
        grid = th.stack([gx, gy], dim=-1)  # (B, V, QD, 2)

        # (5) Nearest sampling from one-hot feature maps
        grid_v = grid.view(B * V, QD, 1, 2)

        sampled = F.grid_sample(
            seg_emb, grid_v, mode='nearest',  
            padding_mode='zeros', align_corners=True
        ) 
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B*V, Q*D, C)
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, C)

        # (6) Mask + Pixel Density Normalization (DOPDN)
        mask_bvqd = mask_bv.view(B, V, Q, self.num_points_in_pillar)

        if self.vote_weight_mode == 'density':
            # 각 픽셀의 투표권 = 1을 depth 기반으로 BEV 셀에 분배 → 같은 픽셀을 공유하는 셀이 많을수록 개별 투표 영향력 감소 (smearing 억제)
            vote_w = self._compute_density_weights(uv, proj_depth, mask_bvqd)  # (B,V,Q,D)
            sampled = sampled * vote_w.unsqueeze(-1)
        else:
            # vote_weight_mode=None: 기존 이진 마스크 (bit-identical 출력 보장)
            sampled = sampled * mask_bvqd.unsqueeze(-1)

        # (7) Weighted Histogram Counting
        count_D = sampled.sum(dim=3)     # (B, V, Q, C) : Z축(Pillar) 방향 가중합
        count_map = count_D.sum(dim=1)   # (B, Q, C)    : View(카메라) 방향 합산

        # Normalize to probability distribution
        denom = count_map.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        f_bev_prob = count_map / denom  # (B, Q, C)

        # (8) Reshape Soft-Histogram to Map format
        seg_bev_prob = f_bev_prob.permute(0, 2, 1).contiguous().view(B, C, self.bev_h, self.bev_w)

        # (9) Map Probability Distribution to Embedding Space via Non-linear Convolution
        # Conv -> SiLU -> Conv 연산을 통해 Spatial 정보를 보존하면서 임베딩 공간으로 매핑
        seg_bev = self.prob_to_emb(seg_bev_prob) # [B, emb_channels, bev_h, bev_w]

        # (10) Add BEV positional embedding
        seg_bev = seg_bev + self.bev_pos_embed

        # (11) Multi-scale encoding
        seg_bev_dict = self.seg_bev_encoder(seg_bev)

        return seg_bev_dict  # {1: [B,C,H,W], 2: [B,C*2,H//2,W//2], 4: [B,C*4,H//4,W//4]}