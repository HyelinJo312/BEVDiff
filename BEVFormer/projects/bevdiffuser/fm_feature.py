import torch.nn as nn
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from timm.models.layers import trunc_normal_
from omegaconf import OmegaConf
import torch.nn.functional as F
from einops import repeat
import os
import torchvision.transforms as T
from collections import OrderedDict
# import lightning as L
from typing import Optional
import math
from transformers import AutoImageProcessor, Dinov2Model, AutoModel
from transformers import CLIPProcessor, CLIPVisionModel

NUM_DECONV = 3
NUM_FILTERS = [32, 32, 32]
DECONV_KERNELS = [2, 2, 2]
VIT_MODEL = 'google/vit-base-patch16-224'



def pad_to_make_square(x):
    y = 255*((x+1)/2)
    y = torch.permute(y, (0,2,3,1))
    bs, _, h, w = x.shape
    if w>h:
        patch = torch.zeros(bs, w-h, w, 3).to(x.device)
        y = torch.cat([y, patch], axis=1)
    else:
        patch = torch.zeros(bs, h, h-w, 3).to(x.device)
        y = torch.cat([y, patch], axis=2)
    return y.to(torch.int)


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts

    
class GetDINOv2Cond(nn.Module):
    """
    DINOv2 condition encoder (HF-only, Tensor-only).

    Inputs:
      - images: torch.Tensor of shape (B, C, H, W) or (B, V, C, H, W), RGB
                values can be in [0,1] or [0,255]

    Returns (preserves view dim if provided):
      - 'cls'   : (B, C_dino) or (B, V, C_dino)          # global embedding (pooler_output)
      - 'cond'  : (B, features) or (B, V, features)      # projected conditioning
      - 'tokens': (B, 1+N, C_dino) or (B, V, 1+N, C_dino)  (optional; last_hidden_state)
    """
    def __init__(
        self,
        encoder: str = 'vitb',     # ['vits', 'vitb', 'vitl'] → small/base/large
        features: int = 256,       # output dim for BEVDiffuser conditioning
        device: str = 'cuda',
        patch: int = 14,
        input_size: int = 518,     
        pretrained: bool = True,
        symmetric_pad: bool = True,
    ):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        self.device = device
        self.input_min = input_size
        self.patch = patch
        self.symmetric_pad = symmetric_pad

        # self.model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.device)
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        # self.model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(self.device)
        # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size  # 384/768/1024

        # ImageNet mean/std buffers (for in-graph normalization)
        # mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        # std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # self.register_buffer('imgnet_mean', mean, persistent=False)
        # self.register_buffer('imgnet_std', std, persistent=False)


    def image_preprocess(self, x):
        """
        x: (N, 3, H, W) float tensor in [0,1] or [0,255]
        -> (x_norm: (N, 3, H2, W2) on self.device, ImageNet normalized,
            Hp, Wp: patch grid size = (H2//patch, W2//patch))
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected (N,3,H,W), got {x.shape}"
        x = x.to(dtype=torch.float32)
        if x.max() > 1.5:      # likely [0,255]
            x = x / 255.0
        x = x.clamp_(0.0, 1.0)

        N, C, H, W = x.shape
        # 종횡비 유지: min(H1, W1) >= input_min
        scale = max(self.input_min / H, self.input_min / W)
        H1, W1 = int(round(H * scale)), int(round(W * scale))

        # 리사이즈
        x = F.interpolate(x, size=(H1, W1),
                          mode='bicubic', align_corners=False)

        # 14 배수 정렬 (ceil) → 패딩
        H2 = (H1 + self.patch - 1) // self.patch * self.patch
        W2 = (W1 + self.patch - 1) // self.patch * self.patch
        pad_h, pad_w = H2 - H1, W2 - W1

        if self.symmetric_pad:
            top = pad_h // 2; bottom = pad_h - top
            left = pad_w // 2; right = pad_w - left
        else:
            top = 0; bottom = pad_h; left = 0; right = pad_w

        x = F.pad(x, (left, right, top, bottom), mode='replicate')

        # x = (x - self.imgnet_mean) / self.imgnet_std
        x = x.to(self.device, non_blocking=True)

        extra_geom = {}
        extra_geom['scale'] = scale
        extra_geom['H2W2'] = (H2, W2)
        extra_geom['padding'] = (top, left)
        extra_geom['patch_size'] = self.patch

        Hp, Wp = H2 // self.patch, W2 // self.patch
        return x, Hp, Wp, extra_geom
    
    def forward(self, images, img_metas, n_layers=4):
        """
        images: (B, 6, C, H, W) or (6, C, H, W) when bs=1
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if images.ndim == 5:
            B, V, C, H, W = images.shape
            x = images.reshape(B * V, C, H, W).contiguous()
        elif images.ndim == 4:
            V, C, H, W = images.shape
            B  = 1
            x = images.reshape(B * V, C, H, W).contiguous()
        else:
            raise ValueError(f"Unexpected tensor shape: {images.shape}")
        
        # x = self._preprocess(x)
        x, Hp, Wp, extra_geom = self.image_preprocess(x)

        # Dinov2 forward
        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_selected = hidden_states[-n_layers:] if n_layers > 0 else [hidden_states[-1]]
        
        feats_out = []
        cls_out = []
        for h in hs_selected:
            cls_tok = h[:, 0]          # (B*V, C_dino)
            tok    = h[:, 1:]          # (B*V, Hp*Wp, C_dino)
            # tok = h[:, 1:, :] 
            cls_tok = cls_tok.view(B, V, self.hidden_dim)                  # (B,V,C)
            tok_seq = tok.view(B, V, Hp * Wp, self.hidden_dim)             # (B,V,N,C)
            feats_out.append(tok_seq)
            cls_out.append(cls_tok)

        last_tok, last_cls = feats_out[-1], cls_out[-1]

        return {
            'feature_type': 'dinov2',
            'features': feats_out,          # list[(B,V,N,C),(B,V,C)]
            'patch_hw': (Hp, Wp),
            'last_cls': last_cls,           # (B, V, C)
            'last_tokens': last_tok,        # (B, V, N, C)
            'img_metas': img_metas,
            'geom': extra_geom
        }


class GetCLIPCond(nn.Module):
    """
    CLIP condition encoder (HF-only, Tensor-only).

    Inputs:
      - images: torch.Tensor of shape (B, C, H, W) or (B, V, C, H, W), RGB
                values can be in [0,1] or [0,255]

    Returns (preserves view dim if provided):
      - 'features': list of tuples per selected layer:
          [(tok_seq: (B,V,N,C), cls_tok: (B,V,C)), ...]
      - 'patch_hw': (Hp, Wp)
      - 'last_cls': (B, V, C)              # global token from last selected layer
      - 'last_tokens': (B, V, N, C)        # patch tokens from last selected layer
      - 'img_metas': passthrough
      - 'clip_geom': dict with preprocess meta (scale, H2W2, padding)
    """
    def __init__(
        self,
        model_id = "openai/clip-vit-base-patch16",  # e.g., "openai/clip-vit-large-patch14-336"
        device='cuda',
        input_size=None,  # if None, use model's vision_config.image_size
        symmetric_pad=True,
    ):
        super().__init__()
        self.device = device
        
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        
        vcfg = self.model.config
        self.hidden_dim = vcfg.hidden_size
        self.patch = vcfg.patch_size                   # CLIP ViT patch
        self.base_image_size = vcfg.image_size         # model's nominal input size (e.g., 224)
        self.input_min = input_size if None else self.base_image_size
        self.symmetric_pad = symmetric_pad

    # @torch.inference_mode()
    def image_preprocess(self, x: torch.Tensor):
        """
        x: (N, 3, H, W) float tensor in [0,1] or [0,255]
        -> (x_norm: (N, 3, H2, W2) on self.device, CLIP normalized,
            Hp, Wp: patch grid size = (H2//patch, W2//patch))
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected (N,3,H,W), got {x.shape}"
        x = x.to(dtype=torch.float32)
        if x.max() > 1.5:  # likely [0,255]
            x = x / 255.0
        x = x.clamp_(0.0, 1.0)

        N, C, H, W = x.shape
        
        # resize (bicubic)
        x = F.interpolate(x, size=(self.input_min, self.input_min), mode='bicubic', align_corners=False)

        x = x.to(self.device, non_blocking=True)
        
        H2 = W2 = self.input_min
        Hp, Wp = H2 // self.patch, W2 // self.patch

        extra_geom = {
            'scale': None,
            'H2W2': (H2, W2),
            'padding': None,
            'target_input_min': self.input_min,
            'patch': self.patch,
        }

        Hp, Wp = H2 // self.patch, W2 // self.patch
        return x, Hp, Wp, extra_geom

    # @torch.inference_mode()
    def forward(self, images: torch.Tensor, img_metas=None, n_layers: int = 4):
        """
        images: (B, V, C, H, W) or (V, C, H, W) (bs=1) or (B, C, H, W)
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if images.ndim == 5:
            B, V, C, H, W = images.shape
            x = images.reshape(B * V, C, H, W).contiguous()
        elif images.ndim == 4:
            # (B, C, H, W) or (V, C, H, W) when bs=1
            if images.shape[0] == 3 and images.shape[1] != 3:
                # guard for ambiguous shapes
                raise ValueError(f"Ambiguous shape {images.shape}; expected (B,3,H,W) or (V,3,H,W).")
            if images.shape[1] == 3:  # (B,3,H,W)
                B = images.shape[0]; V = 1
            else:                     # (V,3,H,W) (assume bs=1)
                B = 1; V = images.shape[0]
            x = images.reshape(B * V, 3, images.shape[-2], images.shape[-1]).contiguous()
        else:
            raise ValueError(f"Unexpected tensor shape: {images.shape}")

        # Preprocess
        x, Hp, Wp, extra_geom = self.image_preprocess(x)

        # CLIP vision forward 
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # list[ (B*V, 1+N, C) ]

        hs_selected = hidden_states[-n_layers:] if n_layers > 0 else [hidden_states[-1]]

        feats_out = []
        for h in hs_selected:
            # h: (B*V, 1+N, C)
            cls_tok = h[:, 0]           # (B*V, C)
            tok    = h[:, 1:]           # (B*V, N, C) with N = Hp*Wp

            cls_tok = cls_tok.view(B, V, self.hidden_dim)          # (B,V,C)
            tok_seq = tok.view(B, V, Hp * Wp, self.hidden_dim)     # (B,V,N,C)
            feats_out.append((tok_seq, cls_tok))

        last_tok, last_cls = feats_out[-1][0], feats_out[-1][1]

        return {
            'feature_type': 'clip',
            'features': feats_out,          # list[(B,V,N,C),(B,V,C)]
            'patch_hw': (Hp, Wp),
            'last_cls': last_cls,           # (B, V, C)
            'last_tokens': last_tok,        # (B, V, N, C)
            'img_metas': img_metas,
            'geom': extra_geom
        }
