import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from torch.utils.checkpoint import checkpoint


@HEADS.register_module()
class BEVFormerOccEncoder(BaseModule):
    """BEVFormer image-to-BEV encoder without a detection decoder/head."""

    def __init__(self,
                 bev_h,
                 bev_w,
                 pc_range,
                 embed_dims,
                 transformer,
                 positional_encoding,
                 init_cfg=None,
                 **kwargs):
        super(BEVFormerOccEncoder, self).__init__(init_cfg=init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.fp16_enabled = False

        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w,
                                          self.embed_dims)
        self._freeze_unused_transformer_branches()

    def _freeze_unused_transformer_branches(self):
        """Freeze the transformer branches occupancy never runs.

        PerceptionTransformer always builds a detection decoder and its
        reference_points layer, but this head only calls get_bev_features().
        Those parameters never receive gradients, which makes DDP abort with
        "parameters that were not used in producing loss". Freezing keeps them
        out of DDP's reducer while leaving init_weights() and checkpoint
        loading untouched.
        """
        for name in ('decoder', 'reference_points'):
            module = getattr(self.transformer, name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def init_weights(self):
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=True,
                given_bev=None,
                **kwargs):
        if given_bev is not None:
            return given_bev

        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device,
                               dtype=dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        return self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
            **kwargs)


@HEADS.register_module()
class Occ3DHead(BaseModule):
    """Lightweight 3D occupancy prediction head for BEV features."""

    def __init__(self,
                 in_channels=256,
                 hidden_channels=64,
                 decoder_channels=128,
                 num_classes=18,
                 occ_z=16,
                 refine_type='conv3d',
                 norm_groups=16,
                 use_checkpoint=True,
                 loss_weight=1.0,
                 init_cfg=None):
        super(Occ3DHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.occ_z = occ_z
        self.refine_type = refine_type
        self.use_checkpoint = use_checkpoint
        self.loss_weight = loss_weight

        self.bev_neck = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.height_lift = nn.Conv2d(hidden_channels,
                                     hidden_channels * occ_z,
                                     kernel_size=1)
        if refine_type == 'conv3d':
            self.volume_refine = nn.Sequential(
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.ReLU(inplace=True),
            )
        elif refine_type == 'mlp':
            self.volume_refine = nn.Sequential(
                nn.Conv3d(hidden_channels, decoder_channels, kernel_size=1),
                nn.Softplus(),
                nn.Conv3d(decoder_channels, hidden_channels, kernel_size=1),
            )
        else:
            raise ValueError(
                f'refine_type must be "mlp" or "conv3d", got {refine_type}')
        self.classifier = nn.Conv3d(hidden_channels, num_classes,
                                    kernel_size=1)

    def _maybe_checkpoint(self, module, feat):
        if self.use_checkpoint and self.training and feat.requires_grad:
            return checkpoint(module, feat)
        return module(feat)

    def forward(self, bev_feat):
        """Forward occupancy logits.

        Args:
            bev_feat (Tensor): BEV feature with shape [B, C, X, Y].

        Returns:
            Tensor: Occupancy logits with shape [B, num_classes, Z, X, Y].
        """
        feat = self.bev_neck(bev_feat)
        feat = self.height_lift(feat)
        b, _, x, y = feat.shape
        feat = feat.view(b, self.hidden_channels, self.occ_z, x, y)
        feat = self._maybe_checkpoint(self.volume_refine, feat)
        return self.classifier(feat)

    @staticmethod
    def _voxels_to_zyx(voxels, spatial_shape, name):
        """Bring Occ3D native [B, X, Y, Z] voxels to logits order [B, Z, X, Y].

        Voxels already stored as [B, Z, X, Y] are passed through unchanged.
        """
        if voxels.dim() != 4:
            raise ValueError(f'{name} must be 4D, got shape {voxels.shape}')

        z, x, y = spatial_shape
        if voxels.shape[1:] == (x, y, z):
            return voxels.permute(0, 3, 1, 2).contiguous()
        if voxels.shape[1:] == (z, x, y):
            return voxels.contiguous()
        raise ValueError(
            f'{name} shape must be [B, X, Y, Z] or [B, Z, X, Y], '
            f'got {tuple(voxels.shape)} for expected ZXY {(z, x, y)}')

    @force_fp32(apply_to=('occ_logits', ))
    def loss(self, occ_logits, gt_occ_semantics, gt_occ_mask=None):
        """Masked cross entropy on camera-visible Occ3D voxels."""
        _, _, z, x, y = occ_logits.shape
        target = self._voxels_to_zyx(
            gt_occ_semantics.long(), (z, x, y), 'gt_occ_semantics')
        mask = None if gt_occ_mask is None else self._voxels_to_zyx(
            gt_occ_mask, (z, x, y), 'gt_occ_mask')

        # Occ3D labels are dense in [0, num_classes); the range check only
        # guards against a corrupted label file reaching the CUDA kernel.
        valid = (target >= 0) & (target < self.num_classes)
        if mask is not None:
            valid = valid & mask.bool()

        if not valid.any():
            return dict(loss_occ=occ_logits.sum() * 0.0)

        logits = occ_logits.permute(0, 2, 3, 4, 1).reshape(-1,
                                                           self.num_classes)
        target = target.reshape(-1)
        valid = valid.reshape(-1)
        loss_occ = F.cross_entropy(logits[valid], target[valid])
        return dict(loss_occ=loss_occ * self.loss_weight)

    @force_fp32(apply_to=('occ_logits', ))
    def get_occ(self, occ_logits):
        """Return predicted labels in Occ3D native [B, X, Y, Z] layout."""
        pred = occ_logits.argmax(dim=1)
        return pred.permute(0, 2, 3, 1).contiguous()
