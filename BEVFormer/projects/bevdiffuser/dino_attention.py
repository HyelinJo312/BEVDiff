import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32

try:
    # mmcv>=1.4 (mmdet3d v0.17.x)
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except Exception:
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


class DINOBevAlignerDeform(nn.Module):
    """
    DINOv2 -> BEV aligned features via deformable attention (single-level, no FPN).

    Inputs:
      - last_tokens: (B, V, N, C_dino)
      - patch_hw:    (Hp, Wp) with Hp*Wp == N
      - img_metas:   list[dict], each dict contains:
          * 'lidar2img': (V, 4, 4)
          * 'img_shape' : ((H, W, 3),)
      - dino_geom:   dict {'scale': float, 'padding': (pt,pl,pr,pb) or (pt,pl), 'H2W2': (H2,W2)}

    Returns:
      - bev_feat_ctx: (B, C_ctx, bev_h, bev_w)
    """
    def __init__(
        self,
        bev_h=50, bev_w=50, cam_view=6,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        c_dino=768, c_ctx=256,
        num_heads=8, num_points=4,
        eps=1e-6, device='cuda'
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.pc_range = pc_range
        self.c_dino, self.c_ctx = c_dino, c_ctx
        self.cam_view = cam_view
        self.eps = eps
        self.device = device

        # View-wise weight (softplus -> positive)
        self.w_view = nn.Parameter(th.zeros(1, cam_view, 1))

        # Single-level projection: (C_dino -> C_ctx)
        self.proj = nn.Conv2d(c_dino, c_ctx, kernel_size=1, bias=False)

        # Fixed BEV query (Q = bev_h * bev_w, dim = c_ctx)
        self.bev_query = nn.Parameter(th.zeros(bev_h * bev_w, c_ctx))
        nn.init.trunc_normal_(self.bev_query, std=0.02)

        # Deformable Cross-Attention (single level)
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=c_ctx,
            num_heads=num_heads,
            num_levels=1,          # <-- single feature level
            num_points=num_points,
            im2col_step=64
        )

    # ------------------------- geometry -------------------------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4, bs=1, device='cuda', dtype=th.float32):
        """Generate normalized 3D reference points for BEV (D x H x W)."""
        zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = th.stack((xs, ys, zs), -1)                               # (D,H,W,3)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)   # (D,Q,3)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)                         # (B,D,Q,3)
        return ref_3d

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def _project_to_cams(self, reference_points, img_metas):
        """
        Project 3D reference points to multi-view image planes.

        Returns:
          uv:       (V, B, Q, D, 2)  - pixel coordinates
          bev_mask: (V, B, Q, D)     - depth > 0 visibility
        """
        # (B,V,4,4)
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas])
        lidar2img = reference_points.new_tensor(lidar2img)

        # Map normalized [0,1] to metric coords using pc_range
        pr = self.pc_range
        ref = reference_points.clone()              # (B,D,Q,3)
        ref[..., 0] = ref[..., 0] * (pr[3] - pr[0]) + pr[0]
        ref[..., 1] = ref[..., 1] * (pr[4] - pr[1]) + pr[1]
        ref[..., 2] = ref[..., 2] * (pr[5] - pr[2]) + pr[2]
        ref = th.cat((ref, th.ones_like(ref[..., :1])), -1)   # (B,D,Q,4)

        # (D,B,N,Q,4,1) x (D,B,N,Q,4,4)
        ref = ref.permute(1, 0, 2, 3)                          # (D,B,Q,4)
        D, B, Q = ref.size()[:3]
        num_cam = lidar2img.size(1)
        ref = ref.view(D, B, 1, Q, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, Q, 1, 1)

        cam = th.matmul(lidar2img.to(th.float32), ref.to(th.float32)).squeeze(-1)  # (D,B,N,Q,4)
        depth = cam[..., 2:3]
        bev_mask = (depth > 1e-5).squeeze(-1)                                      # (D,B,N,Q)
        uv = cam[..., 0:2] / th.clamp(depth, min=1e-5)                              # (D,B,N,Q,2)

        # (V,B,Q,D,2), (V,B,Q,D)
        uv = uv.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0).contiguous()
        return uv, bev_mask

    @staticmethod
    def _tokens_to_fmap(last_tokens, Hp, Wp):
        """(B, V, N, C) -> (B, V, C, Hp, Wp)"""
        B, V, N, C = last_tokens.shape
        return last_tokens.view(B, V, Hp, Wp, C).permute(0, 1, 4, 2, 3).contiguous()

    # --------------------------- forward ---------------------------
    def forward(self, last_tokens, patch_hw, img_metas, dino_geom):
        """
        Produce BEV-aligned feature map using deformable cross-attention over single-level image features.

        Args:
          last_tokens: (B, V, N, C_dino)
          patch_hw:    (Hp, Wp)
          img_metas:   list of dicts
          dino_geom:   preprocessing meta dict

        Returns:
          bev_feat_ctx: (B, C_ctx, bev_h, bev_w)
        """
        assert last_tokens.ndim == 4
        B, V, N, C_d = last_tokens.shape
        Hp, Wp = patch_hw
        assert Hp * Wp == N, f"patch_hw {patch_hw} != N({N})"
        device = last_tokens.device
        dtype = last_tokens.dtype

        # 1) Token map per view (single level)
        fmap = self._tokens_to_fmap(last_tokens, Hp, Wp)              # (B,V,Cd,Hp,Wp)
        f = self.proj(fmap.reshape(B * V, C_d, Hp, Wp))               # (B*V,C_ctx,Hp,Wp)

        # Single-level spatial_shapes and level_start_index
        spatial_shapes = th.as_tensor([[Hp, Wp]], device=device, dtype=th.long)  # (1,2)
        level_start_index = th.as_tensor([0], device=device, dtype=th.long)      # (1,)
        BV = B * V
        value = f.permute(0, 2, 3, 1).reshape(BV, Hp * Wp, self.c_ctx)           # (BV, HW, C)

        # 2) 3D BEV reference points -> camera projection
        Z_bins = int(round(self.pc_range[5] - self.pc_range[2]))
        ref_3d = self._get_reference_points(self.bev_h, self.bev_w, Z=Z_bins,
                                            bs=B, device=device, dtype=dtype)    # (B,D,Q,3)
        uv, bev_mask = self._project_to_cams(ref_3d, img_metas)                  # uv:(V,B,Q,D,2)

        # 3) Pixel coords -> normalized 2D reference per view (depth-weighted mean)
        scale = float(dino_geom['scale'])
        pt, pl = int(dino_geom['padding'][0]), int(dino_geom['padding'][1])
        H2, W2 = int(dino_geom['H2W2'][0]), int(dino_geom['H2W2'][1])

        u = uv[..., 0] * scale + pl   # (V,B,Q,D)
        v = uv[..., 1] * scale + pt
        in_img = (u >= 0) & (u <= (W2 - 1)) & (v >= 0) & (v <= (H2 - 1))
        valid = bev_mask & in_img                                         # (V,B,Q,D)

        refx = th.clamp(u / max(W2 - 1.0, 1.0), 0.0, 1.0)
        refy = th.clamp(v / max(H2 - 1.0, 1.0), 0.0, 1.0)
        w = valid.float().clamp_min(self.eps)
        refx = (refx * w).sum(-1) / w.sum(-1).clamp_min(self.eps)          # (V,B,Q)
        refy = (refy * w).sum(-1) / w.sum(-1).clamp_min(self.eps)          # (V,B,Q)
        ref_2d = th.stack([refx, refy], dim=-1)                             # (V,B,Q,2)

        # 4) Single-level reference points for MSDeformAttn: (BV, Q, 1, 2)
        ref_ms = ref_2d.permute(1, 0, 2, 3).unsqueeze(3)                    # (B,V,Q,1,2)
        ref_ms = ref_ms.contiguous().view(BV, self.bev_h * self.bev_w, 1, 2)

        # 5) Query per (B,V) pair (shared weights)
        query = self.bev_query[None].expand(BV, -1, -1).contiguous()        # (BV,Q,C)

        # 6) Deformable cross-attention (single level)
        out = self.deform_attn(
            query=query, key=None, value=value,
            reference_points=ref_ms,               # (BV,Q,1,2)
            spatial_shapes=spatial_shapes,         # (1,2)
            level_start_index=level_start_index    # (1,)
        )  # (BV, Q, C)

        # 7) View-weighted sum -> (B, C, H, W)
        out = out.view(B, V, self.bev_h * self.bev_w, self.c_ctx)           # (B,V,Q,C)
        wv = F.softplus(self.w_view).expand(B, -1, -1)                        # (B,V,1)
        num = (out * wv.unsqueeze(-1)).sum(dim=1)                             # (B,Q,C)
        den = wv.sum(dim=1).clamp_min(self.eps).unsqueeze(-1)                 # (B,1,1)
        bev_qc = num / den                                                    # (B,Q,C)

        bev_feat_ctx = bev_qc.transpose(1, 2).contiguous().view(B, self.c_ctx, self.bev_h, self.bev_w)
        return bev_feat_ctx
