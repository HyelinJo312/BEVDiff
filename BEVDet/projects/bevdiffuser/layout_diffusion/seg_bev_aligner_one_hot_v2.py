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
        depth_weight_mode='none',  # 'inv_depth' | 'exp_decay' | 'none'
        depth_weight_scale=0.1,     # exp_decay 모드 전용 감쇄 계수
        z_sampling='uniform',       # 'uniform' | 'log' (Z축 샘플링 분포)
        # ---- UV Spread 기반 가중치 옵션 ----
        vote_weight_mode='depth',   # 'depth' (기존) | 'uv_spread' (신규) 투표 가중치 방식 선택
        spread_tau=10.0,            # uv_spread 모드: confidence 전환점 (픽셀 단위)
                                    #   spread < tau → conf ≈ 0 (ray 방향, smearing 위험)
                                    #   spread > tau → conf → 1 (측면 관측, 신뢰 높음)
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
        self.depth_weight_mode = depth_weight_mode
        self.depth_weight_scale = depth_weight_scale
        self.z_sampling = z_sampling
        self.vote_weight_mode = vote_weight_mode
        self.spread_tau = spread_tau

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

    # ---------- Distance-aware depth weighting ----------
    def _compute_depth_weights(self, depth, mask):
        """
        카메라로부터의 투영 거리(depth)를 기반으로 soft weight를 계산한다.

        기하학적 근거:
          Backward projection 시, 가까운 관측(작은 depth)일수록 occlusion에 의해
          가려질 확률이 낮으므로 더 신뢰할 수 있다.
          예) 트럭(10m): w=1/10=0.100 vs 빌딩(50m): w=1/50=0.020
              → 동일 카메라 ray 상에서 트럭이 5배 높은 가중치를 받아
                뒤쪽 셀로의 smearing이 상대적으로 억제됨.

        Args:
            depth: (B, V, Q, D) 카메라 광학축 방향 거리 (미터)
            mask:  (B, V, Q, D) 유효성 불리언 마스크
        Returns:
            weights: (B, V, Q, D) 비음수 가중치, 마스크=False → 0
        """
        if self.depth_weight_mode == 'none':
            return mask.float()

        safe_depth = depth.clamp_min(self.eps)

        if self.depth_weight_mode == 'inv_depth':
            # Perspective geometry의 자연적 가중치: 1/d
            # 가까운 관측에 높은 가중치 → smeared 원거리 투표 억제
            w = 1.0 / safe_depth
        elif self.depth_weight_mode == 'exp_decay':
            # 조정 가능한 지수 감쇄: exp(-α·d)
            # α=0.1일 때 10m→0.37, 50m→0.007 로 급격한 원거리 억제
            w = th.exp(-self.depth_weight_scale * safe_depth)
        else:
            raise ValueError(f"Unknown depth_weight_mode: {self.depth_weight_mode}")

        return w * mask.float()

    # ---------- UV Spread 기반 Per-Camera Confidence Weighting ----------
    def _compute_uv_spread_weights(self, uv, mask):
        """
        Z-샘플들의 투영 UV 좌표 분산(spread)을 카메라별 신뢰도로 사용한다.

        기하학적 근거:
          같은 BEV 셀을 서로 다른 Z-높이(z0..zD)에서 바라볼 때,
          카메라가 BEV 셀을 '측면'에서 보면 각 Z가 이미지의 서로 다른 픽셀에 투영됨
          → UV spread 큼 → 각 Z-샘플이 서로 다른 정보 제공 → 신뢰도 높음.

          반대로 카메라가 BEV 셀을 '광선 방향(ray-along)'으로 바라보면
          모든 Z-샘플이 거의 같은 픽셀에 투영됨 → UV spread ≈ 0
          → 전경 물체 픽셀을 뒤쪽 셀들이 모두 공유하는 smearing 상태
          → 해당 카메라의 투표 신뢰도 낮춤.
        Args:
            uv:   (V, B, Q, D, 2) 원본 픽셀 단위 투영 좌표
            mask: (B, V, Q, D)    유효성 불리언 마스크
        Returns:
            weights: (B, V, Q, D) per-(camera, cell) 가중치; 같은 카메라-셀의 모든 D에 동일 값
        """
        # (V,B,Q,D,2) → (B,V,Q,D,2)
        uv_bvqd = uv.permute(1, 0, 2, 3, 4).contiguous()

        mask_f = mask.float().unsqueeze(-1)                          # (B,V,Q,D,1)
        valid_count = mask_f.sum(dim=3).clamp_min(1.0)               # (B,V,Q,1)

        # D 방향 평균 UV (마스크된 포인트 제외)
        uv_mean = (uv_bvqd * mask_f).sum(dim=3) / valid_count        # (B,V,Q,2)

        # 평균으로부터의 편차
        uv_diff = (uv_bvqd - uv_mean.unsqueeze(3)) * mask_f          # (B,V,Q,D,2)

        # u,v 방향 분산 합산 후 루트 → 픽셀 단위 spread scalar
        uv_var = (uv_diff ** 2).sum(dim=-1).sum(dim=3)               # (B,V,Q)
        uv_var = uv_var / valid_count.squeeze(-1).clamp_min(1.0)
        uv_spread = uv_var.sqrt()                                     # (B,V,Q), 픽셀 단위

        # spread → confidence: 1 - exp(-spread / tau)
        confidence = 1.0 - th.exp(-uv_spread / self.spread_tau)      # (B,V,Q)

        # (B,V,Q) → (B,V,Q,D) broadcast: 같은 카메라-셀의 모든 Z-샘플에 동일 confidence
        weights = confidence.unsqueeze(-1).expand_as(mask).contiguous()
        return weights * mask.float()

    # ---------- BEVFormer-style 3D reference point generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4,
                              dim='3d', bs=1, device='cuda', dtype=th.float32,
                              z_sampling='uniform'):
        if dim == '3d':
            if z_sampling == 'log':
                # Log-spaced Z 샘플링: 지면 근처(낮은 Z)에 샘플을 집중
                # D=4일 때 uniform=[0.06, 0.19, 0.31, 0.44], log≈[0.03, 0.10, 0.22, 0.44]
                # → 하위 50% 구간에 3/4 샘플 집중으로 지면 객체 해상도 향상
                t = th.linspace(0.0, 1.0, num_points_in_pillar, dtype=dtype, device=device)
                zs_norm = (th.exp(t) - 1.0) / (math.e - 1.0)  # [0,1] → [0,1] log-spaced
                zs = (zs_norm * (Z - 1.0) + 0.5) / Z  # [0.5/Z, (Z-0.5)/Z] 범위로 매핑
                zs = zs.view(-1, 1, 1).expand(num_points_in_pillar, H, W)
            else:  # 'uniform' (기본값, 기존 동작)
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
            z_sampling=self.z_sampling,
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

        # (6) Weighted Masking — vote_weight_mode로 방식 선택
        depth_bvqd = proj_depth.permute(1, 0, 2, 3).contiguous()   # (V,B,Q,D) → (B,V,Q,D)
        mask_bvqd  = mask_bv.view(B, V, Q, self.num_points_in_pillar)

        if self.vote_weight_mode == 'uv_spread':
            # UV Spread 기반: Z-샘플들의 투영 좌표 분산이 클수록 신뢰도 높음
            # ray-along 관측(spread≈0) → 낮은 가중치로 smearing 억제
            # 측면 관측(spread 큼)     → 높은 가중치로 원거리 물체 정보 보존
            vote_w = self._compute_uv_spread_weights(uv, mask_bvqd)  # (B,V,Q,D)
        else:
            # 'depth' (기존 동작): 카메라~셀 거리 기반 가중치 (depth_weight_mode 파라미터로 세부 제어)
            vote_w = self._compute_depth_weights(depth_bvqd, mask_bvqd)  # (B,V,Q,D)

        sampled = sampled * vote_w.unsqueeze(-1)  # (B,V,Q,D,1) broadcast → 가중 one-hot

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