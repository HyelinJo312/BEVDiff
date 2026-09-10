import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

'''

Revisied Verison of seg_bev_aligner_one_hot_da3.py

- debug the error in SegEmbedEncoder

'''


class SegEmbedEncoder(nn.Module):
    def __init__(self, num_classes, downsample_factor=2):
        super().__init__()
        self.num_classes = num_classes
        self.downsample_factor = float(downsample_factor)

    def forward(self, seg_id):
        B, V, H, W = seg_id.shape
        seg_id = seg_id.view(B * V, H, W)
        # Keep valid semantic IDs, including sky id 16 when num_classes=16.
        seg_id = seg_id.clamp(min=-1, max=self.num_classes)
        seg_id = torch.where(seg_id == -1, torch.zeros_like(seg_id), seg_id)

        if self.downsample_factor == 1.0:
            seg_id_ds = seg_id.long()
        else:
            # Nearest downsampling preserves discrete class IDs.
            seg_id_ds = F.interpolate(
                            seg_id.unsqueeze(1).float(),
                            scale_factor=1.0 / self.downsample_factor,
                            mode='nearest'
                        ).squeeze(1).long()

        seg_oh = F.one_hot(seg_id_ds, num_classes=self.num_classes+1).float()
        seg_oh = seg_oh.permute(0, 3, 1, 2).contiguous()
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
        # ds=2: Strided Conv (학습 기반 다운샘플 + 채널 확장 X)
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        # ds=4: Strided Conv (학습 기반 다운샘플 + 채널 확장 X)
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
        - seg_downsample_factor: downsampling factor applied before one-hot
          encoding. Use 1 for full-resolution semantic maps and 2 to reproduce
          the previous half-resolution setting.
    Returns:
        - dict {1: ..., 2: ..., 4: ...} multi-scale BEV features
    """

    def __init__(
        self,
        bev_h=64,
        bev_w=64,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        num_classes=15,
        seg_downsample_factor=2,
        emb_channels=256,           # 최종 BEV 임베딩 채널 수 (UNet의 FDN 모델 채널에 맞춤)
        final_dim=(252, 700),       # augmented image size (H, W)
        channel_mult=[1,2,4],
        eps=1e-6,
        v_min_frac=0.0,             # 이미지 상단 배제 비율 [0, 1): e.g. 0.3 → 상위 30%(하늘) 배제
        sky_as_ignore=False,         # sky를 클래스가 아닌 '관측 실패'로 취급
        pillar_z_range=None,        # (z_min, z_max) in lidar frame; None keeps old pc_range-uniform sampling
        # ---- Depth Consistency (FB-BEV) params ----
        depth_consistency_mode=None,  # 'gaussian' | 'bin_linear' | None
        depth_consistency_sigma=2.0,        # Gaussian mode: 허용 오차 σ (meters)
        d_bound=(2.0, 58.0, 0.5),          # bin_linear mode: (start, end, step) in meters
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.num_classes = num_classes
        self.seg_downsample_factor = seg_downsample_factor
        self.emb_channels = emb_channels
        self.final_dim = final_dim
        self.eps = eps
        self.v_min_frac = v_min_frac
        self.depth_consistency_mode = depth_consistency_mode
        self.depth_consistency_sigma = depth_consistency_sigma
        self.d_bound = d_bound
        self.depth_min = d_bound[0]
        self.depth_max = d_bound[1]
        self.depth_step = d_bound[2]
        self.num_depth_bins = int(round((d_bound[1] - d_bound[0]) / d_bound[2]))

        self.pillar_z_range = pillar_z_range
        if pillar_z_range is None:
            self.pillar_z_norm = None
        else:
            zs = th.linspace(float(pillar_z_range[0]), float(pillar_z_range[1]),
                             num_points_in_pillar)
            zn = (zs - pc_range[2]) / (pc_range[5] - pc_range[2])
            assert float(zn.min()) >= 0.0 and float(zn.max()) <= 1.0, \
                f'pillar_z_range {tuple(pillar_z_range)} outside pc_range z ' \
                f'[{pc_range[2]}, {pc_range[5]}]'
            self.register_buffer('pillar_z_norm', zn, persistent=False)
        
        self.sky_as_ignore = sky_as_ignore
        self.sky_id = num_classes if sky_as_ignore else None
        
        # ---- One-hot Segmentation Encoder (설계 간소화, 파라미터 없음) ----
        self.seg_encoder = SegEmbedEncoder(
            num_classes=num_classes,
            downsample_factor=seg_downsample_factor,
        )

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

    # ---------- BEVFormer-style 3D reference point generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4,
                              dim='3d', bs=1, device='cuda', dtype=th.float32,
                              z_norm=None):
        if dim == '3d':
            if z_norm is None:
                zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar,
                                 dtype=dtype, device=device) / Z
            else:
                zs = z_norm.to(device=device, dtype=dtype)
            zs = zs.view(-1, 1, 1).expand(num_points_in_pillar, H, W)
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

    def _align_depth_to_image(self, depth_maps, img_metas, B, V, imgH, imgW):
        """
        FB-BEV Depth Consistency (w_c)
        Bring DA3 depth into the *padded* image frame that `grid_sample` uses.
        """
        dH, dW = depth_maps.shape[-2:]

        contentH, contentW = imgH, imgW
        try:
            # shape = img_metas[0]['img_shape'][0]   # padded size -> no pad applied
            shape = img_metas[0]['ori_shape'][0]     # pre-pad content size
            contentH, contentW = int(shape[0]), int(shape[1])
        except (TypeError, KeyError, IndexError):
            pass
        contentH, contentW = min(contentH, imgH), min(contentW, imgW)

        da3 = F.interpolate(
            depth_maps.reshape(B * V, 1, dH, dW),
            size=(contentH, contentW), mode='bilinear', align_corners=True)
        
        if contentH < imgH or contentW < imgW:
            # zero in the pad region -> w_c = 0, mirroring the ignore label on seg
            da3 = F.pad(da3, (0, imgW - contentW, 0, imgH - contentH))
        return da3

    def _compute_depth_consistency(self, proj_depth, da3_depth):
        """
        Args:
            proj_depth: [B, V, Q, D] — 3D→2D 투영 시의 카메라 Z축 거리 (meters)
            da3_depth:  [B, V, Q, D] — 동일 (u,v) 위치에서 샘플링한 DA3 예측 깊이 (meters)
        Returns:
            w_c: [B, V, Q, D] — depth consistency 가중치, 값 범위 [0, 1]
        """
        if self.depth_consistency_mode == 'gaussian':
            # Gaussian 커널: 깊이 차이가 작을수록 w_c → 1, 클수록 → 0
            # 수식: w_c = exp(-(d_proj - d_da3)^2 / (2 * σ^2))
            diff_sq = (proj_depth - da3_depth).pow(2)
            w_c = th.exp(-diff_sq / (2.0 * self.depth_consistency_sigma ** 2))

        elif self.depth_consistency_mode == 'bin_linear':
            # FB-BEV 원논문 방식: 두 깊이를 이산 bin으로 선형보간하여 내적
            # 각 분포는 인접 2개 bin에만 non-zero → 4가지 인덱스 매칭으로 O(1) 계산
            bin_width = self.depth_step  # d_bound[2]

            # proj_depth 이산화: 연속 깊이 → (lower_bin_idx, upper_bin_idx, frac)
            proj_idx = ((proj_depth - self.depth_min) / bin_width).clamp(0, self.num_depth_bins - 1 - 1e-3)
            proj_lo = proj_idx.floor().long()                                   # [B, V, Q, D]
            proj_hi = (proj_lo + 1).clamp(max=self.num_depth_bins - 1)          # [B, V, Q, D]
            proj_frac = proj_idx - proj_lo.float()                              # [B, V, Q, D]

            # da3_depth 이산화
            da3_idx = ((da3_depth - self.depth_min) / bin_width).clamp(0, self.num_depth_bins - 1 - 1e-3)
            da3_lo = da3_idx.floor().long()                                     # [B, V, Q, D]
            da3_hi = (da3_lo + 1).clamp(max=self.num_depth_bins - 1)            # [B, V, Q, D]
            da3_frac = da3_idx - da3_lo.float()                                 # [B, V, Q, D]

            # 내적: w_c = Σ (w_proj[k] * w_da3[k]) — 4가지 bin 매칭 케이스
            w_c = (
                (proj_lo == da3_lo).float() * (1 - proj_frac) * (1 - da3_frac) +  # lo-lo
                (proj_lo == da3_hi).float() * (1 - proj_frac) * da3_frac       +  # lo-hi
                (proj_hi == da3_lo).float() * proj_frac       * (1 - da3_frac) +  # hi-lo
                (proj_hi == da3_hi).float() * proj_frac       * da3_frac           # hi-hi
            )
            in_range = (
                (proj_depth >= self.depth_min) & (proj_depth <= self.depth_max)
                & (da3_depth >= self.depth_min) & (da3_depth <= self.depth_max)
            )
            w_c = w_c * in_range.float()
        else:
            return th.ones_like(proj_depth)
        w_c = w_c * (da3_depth > 0.5).float()
        return w_c
    
    def _frame_size(self, seg_id, img_metas):
        try:
            pad = img_metas[0]['pad_shape'][0]
            return int(pad[0]), int(pad[1])
        except (TypeError, KeyError, IndexError):
            pass
        if seg_id.dim() == 4:
            return int(seg_id.shape[2]), int(seg_id.shape[3])
        return int(self.final_dim[0]), int(self.final_dim[1])


    def forward(self, seg_id, img_metas, depth_maps=None):
        """
        Args:
            seg_id:     [B, V, H, W] - multi-view segmentation class-id map (e.g. 480×800)
            img_metas:  list[dict] (len=B), each with 'lidar2img': (V,4,4)
            depth_maps: [B, V, dH, dW] - DA3 예측 깊이 (e.g. 448×798), None이면 가중치 미적용
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
            z_norm=self.pillar_z_norm,
        )

        # (3) Project 3D refs to 2D image coords (depth를 보존하여 가중 투표에 활용)
        uv, bev_mask, proj_depth = self.point_sampling(ref_3d, img_metas)  # (V,B,Q,D,2), (V,B,Q,D), (V,B,Q,D)

        # (4) Original pixel → normalised coords for grid_sample
        Q = self.bev_h * self.bev_w
        QD = Q * self.num_points_in_pillar
        
        # imgH, imgW = self.final_dim  # original image size
        # debug image coordinate
        imgH, imgW = self._frame_size(seg_id, img_metas)  # padded frame from pad_shape
        assert (H, W) == (imgH, imgW), \
            f'seg map {(H, W)} != padded frame {(imgH, imgW)}; seg must be padded to pad_shape'

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

        # (6) Depth Consistency Weighting + Masking
        mask_bvqd = mask_bv.view(B, V, Q, self.num_points_in_pillar)  # [B, V, Q, D]

        if depth_maps is not None and self.depth_consistency_mode is not None:
            # (6a) DA3 depth → 이미지 파이프라인과 동일하게 resize + zero-pad
            #      (seg는 ori_shape로 resize 후 pad_shape로 패딩되므로 depth도 동일 경로여야
            #       같은 grid로 샘플링했을 때 같은 픽셀을 읽는다)
            da3 = self._align_depth_to_image(depth_maps, img_metas, B, V, imgH, imgW)  # [B*V,1,imgH,imgW]

            # (6b) 투영된 (u,v) 좌표에서 DA3 depth 샘플링 (seg와 동일한 grid 사용)
            da3_sampled = F.grid_sample(
                da3, grid_v, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )                                                                   # [B*V, 1, QD, 1]
            da3_sampled = da3_sampled.squeeze(1).squeeze(-1)                    # [B*V, QD]
            da3_sampled = da3_sampled.view(B, V, Q, self.num_points_in_pillar)  # [B, V, Q, D]

            # (6c) proj_depth 차원 재배치: (V,B,Q,D) → (B,V,Q,D)
            proj_depth_bvqd = proj_depth.permute(1, 0, 2, 3).contiguous()      # [B, V, Q, D]

            # (6d) Depth consistency 가중치 계산
            w_c = self._compute_depth_consistency(proj_depth_bvqd, da3_sampled) # [B, V, Q, D]

            # (6e) 마스크 × depth consistency 가중치 동시 적용
            sampled = sampled * (mask_bvqd * w_c).unsqueeze(-1)                 # [B, V, Q, D, C]
        else:
            # depth_maps 미제공 시 기존 이진 마스크만 적용 (bit-identical fallback)
            sampled = sampled * mask_bvqd.unsqueeze(-1)

        # (7) Weighted Histogram Counting
        count_D = sampled.sum(dim=3)     # (B, V, Q, C) : Z축(Pillar) 방향 가중합
        count_map = count_D.sum(dim=1)   # (B, Q, C)    : View(카메라) 방향 합산
        
        if self.sky_id is not None:
            sky_ch = self.sky_id
            count_map = count_map.clone()
            count_map[..., 0:1] = count_map[..., 0:1] + count_map[..., sky_ch:sky_ch + 1]
            count_map[..., sky_ch:sky_ch + 1] = 0.0

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
