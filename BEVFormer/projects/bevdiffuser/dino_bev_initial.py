import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

__all__ = ["DINOBevAligner", "DINOLSSFPN"]


def load_pca(path):
    data = np.load(path)
    mu = torch.from_numpy(data["mu"]).float()
    P = torch.from_numpy(data["P"]).float()
    return mu, P


class DINOBevAligner(nn.Module):
    """
    Fixed back-projection BEV backbone for diffusion input.

    This module mirrors the public interface of `DINOLSSFPN`, but replaces
    forward projection / voxel pooling with BEV query back-projection and
    bilinear sampling on PCA-reduced DINO features.
    """

    def __init__(
        self,
        bev_h=128,
        bev_w=128,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        output_channels=80,
        c_dino=768,                 # DINO feature dim
        num_key_frames=2,           # fixed output T for channel dim alignment
        eps=1e-6,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.output_channels = output_channels
        self.c_dino = c_dino
        self.num_key_frames = num_key_frames
        self.eps = eps

        pca_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../pca_ckpts"))
        pca_path = os.path.join(pca_dir, f"pca_sckit_768_to_{self.output_channels}_bevformer.npz")

        mu, P = load_pca(pca_path)
        self.register_buffer("mu", mu)
        self.register_buffer("P", P)

    def apply_pca(self, feats):
        orig_shape = feats.shape[:-1]
        x = feats.reshape(-1, feats.shape[-1])
        x = x - self.mu.to(device=x.device, dtype=x.dtype)
        x = x @ self.P.to(device=x.device, dtype=x.dtype).t()
        return x.reshape(*orig_shape, self.P.size(0))

    # ---------- BEVFormer-style reference generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float32):
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)                 # (D,H,W,3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (D, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)              # (bs, D, H*W, 3)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
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
        lidar2img = torch.stack([
            torch.as_tensor(np.array(m['lidar2img']), dtype=torch.float32, device=reference_points.device)
            for m in img_metas
        ])

        pc_range = self.pc_range
        ref = reference_points.clone()
        ref[..., 0:1] = ref[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref = torch.cat((ref, torch.ones_like(ref[..., :1])), -1)            # (bs, D, Q, 4)

        ref = ref.permute(1, 0, 2, 3)                                  # (D, B, Q, 4)
        D, B, Q = ref.size()[:3]
        num_cam = lidar2img.size(1)

        ref = ref.view(D, B, 1, Q, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # (D,B,N,Q,4,1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, Q, 1, 1)

        cam = torch.matmul(lidar2img.to(torch.float32), ref.to(torch.float32)).squeeze(-1)  # (D,B,N,Q,4)
        eps = 1e-5
        depth = cam[..., 2:3]
        bev_mask = (depth > eps)                                                   # (D,B,N,Q,1)

        uv = cam[..., 0:2] / torch.maximum(depth, torch.ones_like(depth) * eps)          # (D,B,N,Q,2)

        # (V,B,Q,D,2), (V,B,Q,D)
        uv = uv.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()

        # ❌ 기존: allow_tf32 전역 플래그 복원 (비활성화 코드를 제거했으므로 복원도 불필요)
        # th.backends.cuda.matmul.allow_tf32 = allow_tf32
        # th.backends.cudnn.allow_tf32 = allow_tf32_cudnn
        return uv, bev_mask
    
    def _forward_single_sweep(self, sweep_index, img_metas, dino_out):
        """
        Process a single key frame (sweep) and produce a BEV feature map.

        Args:
            sweep_index:     int — index along the T (key frame) dimension.
            sweep_img_metas: list[dict] (len=B) — img_metas for this sweep only.
                             Each dict must contain 'lidar2img': (V, 4, 4).
            dino_out:        dict from GetDINOV2Feat with
                             'last_tokens': (B, T, V, C, Hp, Wp)
        Returns:
            bev_feature: (B, C_out, bev_h, bev_w)
        """
        # (1) Extract this sweep's DINO features: (B, V, C, Hp, Wp)
        dino_feats = dino_out["last_tokens"][:, sweep_index, ...]  # (B, V, C, Hp, Wp)
        dino_geom = dino_out["geom"]
        Hp, Wp = dino_out["patch_hw"]

        # (2) PCA: (B,V,C_dino,Hp,Wp) → channel-last → PCA → channel-first
        tokens = dino_feats.permute(0, 1, 3, 4, 2).contiguous()
        fmap = self.apply_pca(tokens)                    # (B, V, Hp, Wp, C_out)
        fmap = fmap.permute(0, 1, 4, 2, 3).contiguous() # (B, V, C_out, Hp, Wp)
        device = fmap.device
        B, V, C_feat = fmap.shape[:3]

        # (3) Generate BEV reference points and project to camera pixel coords
        Z_bins = int(round(self.pc_range[5] - self.pc_range[2]))
        ref_3d = self._get_reference_points(
                                            self.bev_h, self.bev_w, Z=Z_bins,
                                            num_points_in_pillar=self.num_points_in_pillar,
                                            dim='3d', bs=B, device=device, dtype=fmap.dtype)

        uv, bev_mask = self.point_sampling(ref_3d, img_metas)

        # (3) Map original pixel coords directly to DINO input pixel space (no augmentation)
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
        grid = torch.stack([gx, gy], dim=-1)     # (B, V, QD, 2)

        # (4) bilinear sampling
        fmap_v = fmap.view(B * V, C_feat, Hp, Wp)
        grid_v = grid.view(B * V, QD, 1, 2)

        sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
                                padding_mode='border', align_corners=True)  # (B*V, C, QD, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B*V, QD, C)
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, C_feat)

        # (5) post-norm
        t = sampled.view(-1, C_feat)
        sampled = t.view(B, V, Q, self.num_points_in_pillar, C_feat)

        # (6) pillar mean + view-weighted mean
        mask = mask_bv.view(B, V, Q, self.num_points_in_pillar).unsqueeze(-1).float()  # (B,V,Q,D,1)
        sampled = sampled * mask
        denom_D = mask.sum(dim=3, keepdim=True).clamp_min(self.eps)        # (B,V,Q,1,1)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D                # (B,V,Q,1,C)
        feat_v = feat_v.squeeze(3)                                         # (B,V,Q,C)
        
        # Non-learned view fusion weighted by valid pillar samples per view.
        w = denom_D.squeeze(3)                                             # (B,V,Q,1)
        num = (feat_v * w).sum(dim=1)                                      # (B,Q,C)
        den = w.sum(dim=1).clamp_min(self.eps)                             # (B,Q,1)
        f_bev = num / den                                                  # (B,Q,C_feat)

        # reshape to (B,C_out,H,W)
        bev_feature = f_bev.permute(0,2,1).contiguous().view(B, self.output_channels, self.bev_h, self.bev_w)
        
        return bev_feature


    def forward(self, dino_out, img_metas):
        """
        Args:
            dino_out:   dict from GetDINOV2Feat.
                        'last_tokens': (B, T, V, C, Hp, Wp)
            img_metas:  list[list[dict]] — img_metas[t] = list of B dicts for keyframe t.
                        Each dict contains 'lidar2img': (V, 4, 4).
        Returns:
            BEV features: (B, num_key_frames * output_channels, bev_h, bev_w)
        """
        if dino_out is None:
            raise ValueError("dino_out must be provided for back-projection BEV generation.")

        T = dino_out["num_key_frames"]
        ret_feature_list = []
        for t in reversed(range(T)):
            # img_metas[t] is list of B dicts → pass directly to point_sampling
            ret_feature_list.append(
                self._forward_single_sweep(t, img_metas[t], dino_out))

        result = torch.cat(ret_feature_list, dim=1)  # (B, T*C_out, H, W)

        # Self-replication: pad channels when T < num_key_frames (e.g., inference T=1)
        target_c = self.num_key_frames * self.output_channels
        current_c = result.shape[1]
        if current_c < target_c:
            repeats = target_c // current_c
            result = result.repeat(1, repeats, 1, 1)

        return result

