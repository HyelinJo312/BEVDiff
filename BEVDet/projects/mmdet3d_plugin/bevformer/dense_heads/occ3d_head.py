import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from torch.utils.checkpoint import checkpoint


class ResidualConv3DBlock(nn.Module):
    """Factorized residual block for dense occupancy volumes."""

    def __init__(self, channels, norm_groups):
        super(ResidualConv3DBlock, self).__init__()
        self.conv_z = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.norm_z = nn.GroupNorm(norm_groups, channels)
        self.conv_xy = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm_xy = nn.GroupNorm(norm_groups, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat):
        identity = feat
        feat = self.relu(self.norm_z(self.conv_z(feat)))
        feat = self.norm_xy(self.conv_xy(feat))
        return self.relu(feat + identity)


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
    """3D occupancy head with the legacy and official Occ3D decoders."""

    def __init__(self,
                 in_channels=256,
                 hidden_channels=64,
                 decoder_channels=128,
                 use_decoder_channels=False,
                 num_classes=18,
                 occ_z=16,
                 decoder_type='height_lift',
                 refine_type='conv3d',
                 num_refine_blocks=3,
                 norm_groups=16,
                 occ3d_out_channels=32,
                 use_checkpoint=True,
                 use_camera_mask=True,
                 loss_weight=1.0,
                 class_frequencies=None,
                 free_class_index=17,
                 loss_weight_cfg=None,
                 init_cfg=None):
        super(Occ3DHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.decoder_channels = decoder_channels
        self.use_decoder_channels = use_decoder_channels
        self.num_classes = num_classes
        self.occ_z = occ_z
        self.decoder_type = decoder_type
        self.refine_type = refine_type
        self.num_refine_blocks = num_refine_blocks
        self.occ3d_out_channels = occ3d_out_channels
        self.use_checkpoint = use_checkpoint
        self.use_camera_mask = use_camera_mask
        self.loss_weight = loss_weight
        self.free_class_index = free_class_index
        if not 0 <= self.free_class_index < self.num_classes:
            raise ValueError(f'free_class_index must be in [0, {self.num_classes}), got {self.free_class_index}')

        default_loss_weights = dict(ce=1.0, sem_scal=0.0, geo_scal=0.0, lovasz=0.0)
        if loss_weight_cfg is not None:
            unknown_keys = set(loss_weight_cfg) - set(default_loss_weights)
            if unknown_keys:
                raise KeyError(f'Unsupported occupancy loss weights: {sorted(unknown_keys)}')
            default_loss_weights.update(loss_weight_cfg)
        self.loss_weight_cfg = default_loss_weights

        if class_frequencies is None:
            class_weights = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            frequencies = torch.as_tensor(class_frequencies, dtype=torch.float32)
            if frequencies.numel() != self.num_classes:
                raise ValueError(f'class_frequencies must have {self.num_classes} values, got {frequencies.numel()}')
            if torch.any(frequencies <= 1.0):
                raise ValueError('class_frequencies must contain raw counts greater than 1')
            class_weights = 1.0 / torch.log(frequencies + 0.001)
            class_weights = class_weights / class_weights.mean()
        self.register_buffer('class_weights', class_weights, persistent=False)

        if decoder_type == 'occ3d':
            if in_channels % occ_z != 0:
                raise ValueError(f'Occ3D decoder requires in_channels divisible by occ_z, got {in_channels} and {occ_z}')
            volume_channels = in_channels // occ_z
            self.occ3d_decoder = nn.Sequential(
                nn.Conv3d(volume_channels, occ3d_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(occ3d_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(occ3d_out_channels, occ3d_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(occ3d_out_channels),
                nn.ReLU(inplace=True))
            self.occ3d_predictor = nn.Sequential(
                nn.Linear(occ3d_out_channels, occ3d_out_channels * 2),
                nn.Softplus(),
                nn.Linear(occ3d_out_channels * 2, num_classes))
            for parameter in list(self.occ3d_decoder.parameters()) + list(self.occ3d_predictor.parameters()):
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        elif decoder_type == 'height_lift':
            self.bev_neck = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.ReLU(inplace=True))
            self.height_lift = nn.Conv2d(hidden_channels, hidden_channels * occ_z, kernel_size=1)
            if refine_type == 'conv3d':
                refine_channels = decoder_channels if use_decoder_channels else hidden_channels
                self.volume_refine = nn.Sequential(
                    nn.Conv3d(hidden_channels, refine_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, refine_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(refine_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.ReLU(inplace=True))
            elif refine_type == 'residual_conv3d':
                if num_refine_blocks < 1:
                    raise ValueError(f'num_refine_blocks must be positive, got {num_refine_blocks}')
                self.volume_refine = nn.Sequential(
                    *[ResidualConv3DBlock(hidden_channels, norm_groups) for _ in range(num_refine_blocks)])
            elif refine_type == 'mlp':
                self.volume_refine = nn.Sequential(
                    nn.Conv3d(hidden_channels, decoder_channels, kernel_size=1),
                    nn.Softplus(),
                    nn.Conv3d(decoder_channels, hidden_channels, kernel_size=1))
            else:
                raise ValueError(f'refine_type must be "mlp", "conv3d", or "residual_conv3d", got {refine_type}')
            self.classifier = nn.Conv3d(hidden_channels, num_classes, kernel_size=1)
        else:
            raise ValueError(f'decoder_type must be "height_lift" or "occ3d", got {decoder_type}')

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
        if self.decoder_type == 'occ3d':
            b, c, x, y = bev_feat.shape
            feat = bev_feat.reshape(b, c // self.occ_z, self.occ_z, x, y)
            feat = self._maybe_checkpoint(self.occ3d_decoder, feat)
            logits = self.occ3d_predictor(feat.permute(0, 2, 3, 4, 1))
            return logits.permute(0, 4, 1, 2, 3).contiguous()

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

    @staticmethod
    def _negative_log_score(score, eps=1e-5):
        return -torch.log(score.clamp(min=eps, max=1.0))

    def _semantic_scaling_loss(self, probabilities, target):
        losses = []
        for class_index in range(self.num_classes):
            if class_index == self.free_class_index:
                continue
            class_target = (target == class_index).float()
            positive_count = class_target.sum()
            if positive_count == 0:
                continue

            class_probability = probabilities[:, class_index]
            intersection = (class_probability * class_target).sum()
            precision = intersection / (class_probability.sum() + 1e-5)
            recall = intersection / (positive_count + 1e-5)
            negative_target = 1.0 - class_target
            specificity = ((1.0 - class_probability) * negative_target).sum() / (negative_target.sum() + 1e-5)
            losses.append(self._negative_log_score(precision) + self._negative_log_score(recall)
                          + self._negative_log_score(specificity))
        return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0

    def _geometric_scaling_loss(self, probabilities, target):
        free_probability = probabilities[:, self.free_class_index]
        occupied_probability = 1.0 - free_probability
        occupied_target = (target != self.free_class_index).float()
        free_target = 1.0 - occupied_target
        intersection = (occupied_probability * occupied_target).sum()
        precision = intersection / (occupied_probability.sum() + 1e-5)
        recall = intersection / (occupied_target.sum() + 1e-5)
        specificity = (free_probability * free_target).sum() / (free_target.sum() + 1e-5)
        return (self._negative_log_score(precision) + self._negative_log_score(recall)
                + self._negative_log_score(specificity))

    @staticmethod
    def _lovasz_gradient(sorted_target):
        target_sum = sorted_target.sum()
        intersection = target_sum - sorted_target.cumsum(0)
        union = target_sum + (1.0 - sorted_target).cumsum(0)
        gradient = 1.0 - intersection / union
        if sorted_target.numel() > 1:
            gradient[1:] = gradient[1:] - gradient[:-1]
        return gradient

    def _lovasz_softmax_loss(self, probabilities, target):
        losses = []
        for class_index in range(self.num_classes):
            class_target = (target == class_index).float()
            if class_target.sum() == 0:
                continue
            errors = (class_target - probabilities[:, class_index]).abs()
            sorted_errors, permutation = torch.sort(errors, descending=True)
            sorted_target = class_target[permutation]
            losses.append(torch.dot(sorted_errors, self._lovasz_gradient(sorted_target)))
        return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0

    @force_fp32(apply_to=('occ_logits', ))
    def loss(self, occ_logits, gt_occ_semantics, gt_occ_mask=None):
        """Calculate occupancy losses, optionally on camera-visible voxels."""
        _, _, z, x, y = occ_logits.shape
        target = self._voxels_to_zyx(gt_occ_semantics.long(), (z, x, y), 'gt_occ_semantics')
        mask = None
        if self.use_camera_mask and gt_occ_mask is not None:
            mask = self._voxels_to_zyx(gt_occ_mask, (z, x, y), 'gt_occ_mask')

        # Occ3D labels are dense in [0, num_classes); the range check only
        # guards against a corrupted label file reaching the CUDA kernel.
        valid = (target >= 0) & (target < self.num_classes)
        if mask is not None:
            valid = valid & mask.bool()

        if not valid.any():
            zero = occ_logits.sum() * 0.0
            losses = dict(loss_occ=zero)
            if self.loss_weight_cfg['sem_scal']:
                losses['loss_occ_sem_scal'] = zero
            if self.loss_weight_cfg['geo_scal']:
                losses['loss_occ_geo_scal'] = zero
            if self.loss_weight_cfg['lovasz']:
                losses['loss_occ_lovasz'] = zero
            return losses

        logits = occ_logits.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        target = target.reshape(-1)
        valid = valid.reshape(-1)
        logits = logits[valid]
        target = target[valid]
        loss_occ = F.cross_entropy(logits, target, weight=self.class_weights.to(logits))
        losses = dict(loss_occ=loss_occ * self.loss_weight_cfg['ce'] * self.loss_weight)
        if any(self.loss_weight_cfg[name] for name in ('sem_scal', 'geo_scal', 'lovasz')):
            probabilities = F.softmax(logits, dim=1)
            if self.loss_weight_cfg['sem_scal']:
                losses['loss_occ_sem_scal'] = (self._semantic_scaling_loss(probabilities, target)
                                                * self.loss_weight_cfg['sem_scal'] * self.loss_weight)
            if self.loss_weight_cfg['geo_scal']:
                losses['loss_occ_geo_scal'] = (self._geometric_scaling_loss(probabilities, target)
                                                * self.loss_weight_cfg['geo_scal'] * self.loss_weight)
            if self.loss_weight_cfg['lovasz']:
                losses['loss_occ_lovasz'] = (self._lovasz_softmax_loss(probabilities, target)
                                             * self.loss_weight_cfg['lovasz'] * self.loss_weight)
        return losses

    @force_fp32(apply_to=('occ_logits', ))
    def get_occ(self, occ_logits):
        """Return predicted labels in Occ3D native [B, X, Y, Z] layout."""
        pred = occ_logits.argmax(dim=1)
        return pred.permute(0, 2, 3, 1).contiguous()
