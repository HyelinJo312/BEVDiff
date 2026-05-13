import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from transformers import CLIPTokenizer, CLIPTextModel


# Class ID -> text prompts mapping (merged classes use multiple prompts, averaged)
SEG_CLASS_TEXTS = {
    0: ['background'],
    1: ['sedan'],
    2: ['highway'],
    3: ['bus'],
    4: ['truck'],
    5: ['terrain'],
    6: ['tree'],
    7: ['sidewalk'],
    8: ['bicycle', 'bicyclist'],
    9: ['barrier', 'barricade'],
    10: ['person', 'pedestrian'],
    11: ['building', 'bridge', 'pole', 'billboard', 'light', 'ashbin'],
    12: ['motorcycle', 'motorcyclist'],
    13: ['crane'],
    14: ['trailer'],
    15: ['cone'],
    16: ['sky'],
}


def build_clip_embeddings():
    print('[SegBEVAligner] Building CLIP embeddings (first run only)...')
    tokenizer = CLIPTokenizer.from_pretrained(
        'stabilityai/stable-diffusion-2-1', subfolder='tokenizer'
    )
    text_encoder = CLIPTextModel.from_pretrained(
        'stabilityai/stable-diffusion-2-1', subfolder='text_encoder'
    )
    text_encoder.requires_grad_(False)

    embeddings = []
    for class_id in range(17):  # 0-16
        texts = SEG_CLASS_TEXTS[class_id]
        tokens = tokenizer(
            texts,
            max_length=tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).input_ids
        with torch.no_grad():
            clip_emb = text_encoder(tokens)[1]  # (N_texts, 1024), pooled output
        embeddings.append(clip_emb.mean(dim=0))  # average for merged classes

    result = torch.stack(embeddings)  # (17, 1024)
    del text_encoder, tokenizer
    print('[SegBEVAligner] CLIP embeddings ready.')
    return result


class SegEmbedEncoder(nn.Module):
    """
    Input:  [B, V, H, W] segmentation ids  (e.g. 480x800)
    Output: [B*V, out_channels, H//2, W//2]  (e.g. 240x400)
    """
    def __init__(self, num_classes, embed_dim, out_channels, clip_dim=1024):
        super().__init__()
        self.embed = nn.Embedding(num_classes + 1, embed_dim)
        # 2-layer MLP: 1024 → 256 → 64 (점진적 압축으로 semantic 정보 보존)
        clip_hidden = max(clip_dim // 4, embed_dim * 2)  # 256
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, clip_hidden),
            nn.SiLU(),
            nn.Linear(clip_hidden, embed_dim),
        )
        self.downsample_factor = 2

        self.encoder = nn.Sequential(
            # Stage 1: full res, lightweight channels -> boundary context
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            # Stage 2: stride-2 downsample + channel expansion
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            # Stage 3: half res, full channels -> context refinement
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, seg_id, clip_embeddings):
        """
        Args:
            seg_id: (B*V, H, W) int tensor
            clip_embeddings: (17, 1024) frozen CLIP text embeddings
        """
        B_V, H, W = seg_id.shape

        seg_id = seg_id.clamp(min=-1, max=16)
        seg_id = torch.where(seg_id == -1, torch.zeros_like(seg_id), seg_id)

        clip_feat = self.clip_proj(clip_embeddings)          # (17, embed_dim)
        combined = self.embed.weight + clip_feat             # (17, embed_dim)
        seg_emb = F.embedding(seg_id.long(), combined)       # (B*V, H, W, embed_dim)

        seg_emb = seg_emb.permute(0, 3, 1, 2).contiguous()  # (B*V, embed_dim, H, W)
        seg_emb = self.encoder(seg_emb)  # (B*V, out_channels, H//2, W//2)

        return seg_emb


class SegTextCrossAttention(nn.Module):
    """BEV-level cross-attention: each BEV position attends to CLIP class embeddings."""
    def __init__(self, bev_channels, clip_dim=1024, num_heads=4):
        super().__init__()
        self.clip_proj = nn.Linear(clip_dim, bev_channels)
        self.cross_attn = nn.MultiheadAttention(
            bev_channels, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(bev_channels)

    def forward(self, seg_bev, clip_embeddings):
        """
        Args:
            seg_bev: (B, C, H, W)
            clip_embeddings: (17, 1024) frozen CLIP text embeddings
        Returns:
            (B, C, H, W) enriched BEV features
        """
        B, C, H, W = seg_bev.shape

        q = seg_bev.flatten(2).transpose(1, 2)  # (B, H*W, C)
        kv = self.clip_proj(clip_embeddings)     # (17, C)
        kv = kv.unsqueeze(0).expand(B, -1, -1)  # (B, 17, C)

        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, H*W, C)
        attn_out = self.norm(attn_out)

        seg_bev = seg_bev + attn_out.transpose(1, 2).view(B, C, H, W)
        return seg_bev


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

        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c4, kernel_size=2, stride=2),
            nn.SiLU(),
        )

    def forward(self, bev_ctx):
        s1 = bev_ctx          # (B, C, 50, 50)
        s2 = self.down1(s1)   # (B, C*2, 25, 25)
        s4 = self.down2(s2)   # (B, C*4, 12, 12)
        return {1: s1, 2: s2, 4: s4}


class SegBEVAligner(nn.Module):
    """
    BEVFormer-style IPM for multi-view segmentation ID map -> BEV feature.

    v3 changes vs v2:
      1. CLIP text embedding: per-pixel semantic enrichment in SegEmbedEncoder
      2. SegTextCrossAttention: BEV-level cross-attention with CLIP class tokens

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
        final_dim=(252, 700),
        channel_mult=[1, 2, 4],
        clip_dim=1024,
        use_clip_text=False,
        use_text_cross_attention=False,
        text_cross_attn_heads=4,
        eps=1e-6,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.emb_channels = emb_channels
        self.final_dim = final_dim
        self.eps = eps
        self.use_clip_text = use_clip_text
        self.use_text_cross_attention = use_text_cross_attention

        # ---- CLIP text embeddings (frozen, lazy: computed on first forward if not in checkpoint) ----
        if self.use_clip_text or self.use_text_cross_attention:
            self.register_buffer('clip_embeddings', None)  # filled on first forward or restored from checkpoint

        # ---- Segmentation embedding encoder ----
        self.seg_encoder = SegEmbedEncoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            out_channels=self.emb_channels,
            clip_dim=clip_dim if self.use_clip_text else 0,
        )

        # ---- Feature normalization ----
        self.post_norm = nn.GroupNorm(num_groups=8, num_channels=self.emb_channels)

        # ---- Per-view learnable weights ----
        self._w_view = nn.Parameter(th.zeros(1, cam_view, 1))

        # ---- Learnable 2D BEV positional embedding ----
        self.bev_pos_embed = nn.Parameter(th.zeros(1, self.emb_channels, bev_h, bev_w))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        # ---- BEV-level cross-attention with CLIP class tokens ----
        if self.use_text_cross_attention:
            self.text_cross_attn = SegTextCrossAttention(
                bev_channels=self.emb_channels,
                clip_dim=clip_dim,
                num_heads=text_cross_attn_heads,
            )

        # ---- Multi-scale BEV encoder ----
        self.seg_bev_encoder = SegBEVEncoder(
            in_channels=self.emb_channels,
            channel_mult=channel_mult,
        )

    # ---------- BEVFormer-style 3D reference point generation ----------
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

    # ---------- 3D -> 2D camera projection ----------
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
        ref = th.cat((ref, th.ones_like(ref[..., :1])), -1)  # (bs, D, Q, 4)

        ref = ref.permute(1, 0, 2, 3)  # (D, B, Q, 4)
        D, B, Q = ref.size()[:3]
        num_cam = lidar2img.size(1)

        ref = ref.view(D, B, 1, Q, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, Q, 1, 1)

        cam = th.matmul(lidar2img.to(th.float32), ref.to(th.float32)).squeeze(-1)
        eps = 1e-5
        depth = cam[..., 2:3]
        bev_mask = (depth > eps)

        uv = cam[..., 0:2] / th.maximum(depth, th.ones_like(depth) * eps)

        uv = uv.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()

        return uv, bev_mask

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
        C = self.emb_channels

        # (1) Lazy-init CLIP embeddings: compute once on first forward, restored from checkpoint thereafter
        if (self.use_clip_text or self.use_text_cross_attention) and self.clip_embeddings is None:
            clip_emb = build_clip_embeddings().to(device)
            self.clip_embeddings = clip_emb  # assigned into buffer in-place

        # Embed segmentation IDs -> feature maps (with CLIP text enrichment)
        seg_id_flat = seg_id.view(B * V, H, W)
        clip_emb = self.clip_embeddings if self.use_clip_text else None
        seg_emb = self.seg_encoder(seg_id_flat, clip_emb)  # [B*V, C, fH, fW]

        # (2) Generate BEV 3D reference points
        Z_bins = int(round(self.pc_range[5] - self.pc_range[2]))
        ref_3d = self._get_reference_points(
            self.bev_h, self.bev_w, Z=Z_bins,
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d', bs=B, device=device, dtype=seg_emb.dtype
        )

        # (3) Project 3D refs to 2D image coords
        uv, bev_mask = self.point_sampling(ref_3d, img_metas)

        # (4) Pixel coords -> normalised coords for grid_sample
        Q = self.bev_h * self.bev_w
        QD = Q * self.num_points_in_pillar
        imgH, imgW = self.final_dim
        ds = self.seg_encoder.downsample_factor
        fH, fW = imgH // ds, imgW // ds

        uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)

        u = uv_flat[..., 0]
        v = uv_flat[..., 1]

        valid_in = (u >= 0) & (u <= (imgW - 1)) & (v >= 0) & (v <= (imgH - 1))
        bev_mask_flat = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD)
        mask_bv = bev_mask_flat & valid_in

        u_feat = u / ds
        v_feat = v / ds
        gx = 2.0 * (u_feat / (fW - 1.0)) - 1.0
        gy = 2.0 * (v_feat / (fH - 1.0)) - 1.0
        grid = th.stack([gx, gy], dim=-1)

        # (5) Bilinear sampling from seg feature maps
        grid_v = grid.view(B * V, QD, 1, 2)
        sampled = F.grid_sample(
            seg_emb, grid_v, mode='bilinear',
            padding_mode='border', align_corners=True
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        sampled = sampled.view(B, V, Q, self.num_points_in_pillar, C)

        # (6) Pillar mean + view-weighted mean
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

        # (7) Reshape to (B, C, bev_h, bev_w)
        seg_bev = f_bev.permute(0, 2, 1).contiguous().view(B, C, self.bev_h, self.bev_w)

        # (8) GroupNorm + BEV positional embedding
        seg_bev = self.post_norm(seg_bev)
        seg_bev = seg_bev + self.bev_pos_embed

        # (9) BEV-level cross-attention with CLIP class tokens
        if self.use_text_cross_attention:
            seg_bev = self.text_cross_attn(seg_bev, self.clip_embeddings)

        # (10) Multi-scale encoding
        seg_bev_dict = self.seg_bev_encoder(seg_bev)

        return seg_bev_dict
