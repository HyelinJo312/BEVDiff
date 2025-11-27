# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from LayoutDiffusion
#   (https://github.com/ZGCTroy/LayoutDiffusion)
# Copyright (c) 2023 LayoutDiffusion authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from abc import abstractmethod
import os
import safetensors
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
from projects.bevdiffuser.ldm.modules.attention import SpatialTransformer
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version

from .dino_utils import (
    build_bev_xy_grid,
    project_lidar2img,
    uv_to_patch_grid,
    masked_mean
)

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

class SiLU(nn.Module):  # export-friendly version of SiLU()
    @staticmethod
    def forward(x):
        return x * th.sigmoid(x)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, dino_cond=None, cond_kwargs=None):
        extra_output = None
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, (AttentionBlock, ObjectAwareCrossAttention)):
                x, extra_output = layer(x, cond_kwargs)
            elif isinstance(layer, DINOCrossAttention):
                x = layer(x, dino_cond)
            else:
                x = layer(x)
        return x, extra_output


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, out_size=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.out_size = out_size
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.out_size is None:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(
                    x, (x.shape[2], self.out_size, self.out_size), mode="nearest"
                )
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=self.out_size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            out_size=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, out_size=out_size)
            self.x_upd = Upsample(channels, False, dims, out_size=out_size)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=False
    ):
        super().__init__()
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.randn(channels // self.num_heads, resolution ** 2) / channels ** 0.5)  # [C,L1]
        else:
            self.positional_embedding = None

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)

        self.qkv = conv_nd(1, channels, channels * 3, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.encoder_channels = encoder_channels
        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs=None):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        qkv = self.qkv(self.norm(x))  # N x 3C x L1, 其中L1=H*W
        if cond_kwargs is not None and self.encoder_channels is not None:
            kv_for_encoder_out = self.encoder_kv(cond_kwargs['xf_out'])  # xf_out: (N x encoder_channels x L2) -> (N x 2C x L2), 其中L2=max_obj_num
            h = self.attention(qkv, kv_for_encoder_out, positional_embedding=self.positional_embedding)
        else:
            h = self.attention(qkv, positional_embedding=self.positional_embedding)
        h = self.proj_out(h)
        output = (x + h).reshape(b, c, *spatial)

        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({
                'type': self.type,
                'ds': self.ds,
                'resolution': self.resolution,
                'num_heads': self.num_heads,
                'num_channels': self.channels,
                'image_query_embeddings': qkv[:, :self.channels, :].detach(),  # N x C x L1
            })
            if cond_kwargs is not None:
                extra_output.update({
                    'layout_key_embeddings': kv_for_encoder_out[:, : self.channels, :].detach()  # N x C x L2
                })

        return output, extra_output


class ObjectAwareCrossAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=True,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False
    ):
        super().__init__()
        self.norm_for_obj_embedding=None
        self.norm_first = norm_first
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.use_key_padding_mask=use_key_padding_mask
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        assert self.use_positional_embedding

        self.use_checkpoint = use_checkpoint

        self.qkv_projector = conv_nd(1, channels, 3 * channels, 1)
        self.norm_for_qkv = normalization(channels)

        if encoder_channels is not None:
            self.encoder_channels= encoder_channels
            self.layout_content_embedding_projector = conv_nd(1, encoder_channels, channels * 2, 1)
            self.layout_position_embedding_projector = conv_nd(1, encoder_channels, int(channels * self.channels_scale_for_positional_embedding), 1)
            if self.norm_first:
                if norm_for_obj_embedding:
                    self.norm_for_obj_embedding = normalization(encoder_channels)
                self.norm_for_obj_class_embedding = normalization(encoder_channels)
                self.norm_for_layout_positional_embedding = normalization(encoder_channels)
                self.norm_for_image_patch_positional_embedding = normalization(encoder_channels)
            else:
                self.norm_for_obj_class_embedding = normalization(encoder_channels)
                self.norm_for_layout_positional_embedding = normalization(int(channels * self.channels_scale_for_positional_embedding))
                self.norm_for_image_patch_positional_embedding = normalization(int(channels * self.channels_scale_for_positional_embedding))

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        qkv = self.qkv_projector(self.norm_for_qkv(x))  # N x 3C x L1, 其中L1=H*W
        bs, C, L1, L2 = qkv.shape[0], self.channels, qkv.shape[2], cond_kwargs['obj_bbox_embedding'].shape[-1]  # L2=300 (# of objects)

        # positional embedding for image patch
        if self.norm_first:
            image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(cond_kwargs['image_patch_bbox_embedding_for_resolution{}'.format(self.resolution)])  # (N, encoder_channels, L1)
            image_patch_positional_embedding = self.layout_position_embedding_projector(image_patch_positional_embedding)  # N x C * channels_scale_for_positional_embedding x L1, 其中L1=H*W
        else:
            image_patch_positional_embedding = self.layout_position_embedding_projector(
                cond_kwargs['image_patch_bbox_embedding_for_resolution{}'.format(self.resolution)]
            )  # N x C * channels_scale_for_positional_embedding x L1, 其中L1=H*W
            image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(image_patch_positional_embedding)  # (N, C * channels_scale_for_positional_embedding, L1)
        image_patch_positional_embedding = image_patch_positional_embedding.reshape(bs * self.num_heads, int(C * self.channels_scale_for_positional_embedding) // self.num_heads, L1)  # (N * num_heads, C * channels_scale_for_positional_embedding // num_heads, L1)

        # content embedding for image patch
        q_image_patch_content_embedding, k_image_patch_content_embedding, v_image_patch_content_embedding = qkv.split(C, dim=1)  # 3 x (N , C, L1)
        q_image_patch_content_embedding = q_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)  # (N // num_heads, C // num_heads, L1)
        k_image_patch_content_embedding = k_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)  # (N // num_heads, C // num_heads, L1)
        v_image_patch_content_embedding = v_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)  # (N // num_heads, C // num_heads, L1)

        # embedding for image patch
        q_image_patch = torch.cat([q_image_patch_content_embedding, image_patch_positional_embedding], dim=1)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1)
        k_image_patch = torch.cat([k_image_patch_content_embedding, image_patch_positional_embedding], dim=1)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1)
        v_image_patch = v_image_patch_content_embedding  # (N // num_heads, C // num_heads, L1)

        # positional embedding for layout
        if self.norm_first:
            layout_positional_embedding = self.norm_for_layout_positional_embedding(cond_kwargs['obj_bbox_embedding'])  # (N, encoder_channels, L2)
            layout_positional_embedding = self.layout_position_embedding_projector(layout_positional_embedding)  # N x C*channels_scale_for_positional_embedding x L2
        else:
            layout_positional_embedding = self.layout_position_embedding_projector(cond_kwargs['obj_bbox_embedding'])  # N x C*channels_scale_for_positional_embedding x L2
            layout_positional_embedding = self.norm_for_layout_positional_embedding(layout_positional_embedding)  # (N, C * channels_scale_for_positional_embedding, L2)
        layout_positional_embedding = layout_positional_embedding.reshape(bs * self.num_heads, int(C * self.channels_scale_for_positional_embedding) // self.num_heads, L2)  # (N // num_heads, channels_scale_for_positional_embedding * C // num_heads, L2)

        # content embedding for layout
        if self.norm_for_obj_embedding is not None:
            layout_content_embedding = (self.norm_for_obj_embedding(cond_kwargs['xf_out']) + self.norm_for_obj_class_embedding(cond_kwargs['obj_class_embedding'])) / 2
        else:
            layout_content_embedding = (cond_kwargs['xf_out'] + self.norm_for_obj_class_embedding(cond_kwargs['obj_class_embedding'])) / 2
        k_layout_content_embedding, v_layout_content_embedding = self.layout_content_embedding_projector(layout_content_embedding).split(C, dim=1)  # 2 x (N x C x L2)
        k_layout_content_embedding = k_layout_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L2)  # (N // num_heads, C // num_heads, L2)
        v_layout_content_embedding = v_layout_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L2)  # (N // num_heads, C // num_heads, L2)

        # embedding for layout
        k_layout = torch.cat([k_layout_content_embedding, layout_positional_embedding], dim=1)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L2)
        v_layout = v_layout_content_embedding  # (N // num_heads, C // num_heads, L2)

        #  mix embedding for cross attention
        k_mix = th.cat([k_image_patch, k_layout], dim=2)  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1+L2)
        v_mix = th.cat([v_image_patch, v_layout], dim=2)  # (N // num_heads, 1 * C // num_heads, L1+L2)

        if self.use_key_padding_mask:
            key_padding_mask = torch.cat(
                [
                    torch.zeros((bs, L1), device=cond_kwargs['key_padding_mask'].device).bool(),  # (N, L1)
                    cond_kwargs['key_padding_mask']  # (N, L2)
                ],
                dim=1
            )  # (N, L1+L2)
            print(cond_kwargs['key_padding_mask'])

        scale = 1 / math.sqrt(math.sqrt(int((1+self.channels_scale_for_positional_embedding) * C) // self.num_heads))
        attn_output_weights = th.einsum(
            "bct,bcs->bts", q_image_patch * scale, k_mix * scale
        )  # More stable with f16 than dividing afterwards, (N x num_heads, L1, L1+L2)

        attn_output_weights = attn_output_weights.view(bs, self.num_heads, L1, L1 + L2)

        if self.use_key_padding_mask:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1+L2)
                float('-inf'),
            )
        attn_output_weights = attn_output_weights.view(bs * self.num_heads, L1, L1 + L2)

        attn_output_weights = th.softmax(attn_output_weights.float(), dim=-1).type(attn_output_weights.dtype)  # (N x num_heads, L1, L1+L2)

        attn_output = th.einsum("bts,bcs->bct", attn_output_weights, v_mix)  # (N x num_heads, C // num_heads, L1)
        attn_output = attn_output.reshape(bs, C, L1)  # (N, C, L1)

        #
        h = self.proj_out(attn_output)

        output = (x + h).reshape(b, c, *spatial)  # B, C, H, W

        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({
                'type': self.type,
                'ds': self.ds,
                'resolution': self.resolution,
                'num_heads': self.num_heads,
                'num_channels': self.channels,
                'image_query_embeddings': image_patch_positional_embedding.detach().view(bs, -1, L1),  # N x C x L1
                # 'image_query_embeddings': qkv[:, :self.channels, :].detach(),  # N x C x L1
            })
            if cond_kwargs is not None:
                extra_output.update({
                    'layout_key_embeddings': layout_positional_embedding.detach().view(bs, -1, L2)  # N x C x L2

                    # 'layout_key_embeddings': kv_for_encoder_out[:, : self.channels, :].detach()  # N x C x L2
                })

        return output, extra_output


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None, positional_embedding=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Q_T, K_T, and V_T.
        :param encoder_kv: an [N x (H * 2 * C) x S] tensor of K_E, and V_E.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if positional_embedding is not None:
            q = q + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]
            k = k + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]

        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class DINOContextAdapter(nn.Module):
    def __init__(self, 
                 c_in=768, 
                 c_emb=1024,
                 num_views=6,
                 dropout=0.0,
                 ln_first=True,
                 ln_after=True,
                 use_cam_embed=True,
                 temperature=1.0):
        super().__init__()
        self.num_views = num_views
        self.use_cam_embed = use_cam_embed
        self.temperature = temperature

        self.ln_first = nn.LayerNorm(c_in) if ln_first else None

        # camera embedding (learned)
        if use_cam_embed:
            self.cam_embed = nn.Embedding(num_views, c_in)
            nn.init.normal_(self.cam_embed.weight, std=0.02)
        else:
            self.register_parameter("cam_embed", None)

        # optional per-view bias (helps learning)
        self.view_bias = nn.Parameter(th.zeros(num_views))

        # projection to emb dim
        # self.proj = nn.Sequential(
        #     nn.Linear(c_in, c_emb),
        #     nn.GELU(),
        #     nn.Linear(c_emb, c_emb),
        # )
        self.proj = nn.Linear(c_in, c_emb)
  
        # self.ln_after = nn.LayerNorm(c_emb) if ln_after else None

    def forward(self, context, cam_ids=None):
        """
        context: (B, V, C_in)  or (B, C_in) -> treated as V=1
        cam_ids: (B, V) long indices in [0, num_views-1] (optional; if None, 0..V-1)
        """
        if context.ndim == 2:
            context = context.unsqueeze(1)  # (B,1,C)
        elif context.ndim != 3:
            raise ValueError(f"context must be (B,C) or (B,V,C), got {context.shape}")

        B, V, C = context.shape
        x = context
        if self.ln_first is not None:
            x = self.ln_first(x)  # LN over C

        # ----- view weighting -----
        if self.use_cam_embed:
            cam_embed = self.cam_embed(cam_ids)               # (B, V, C)
            # dot-product score with temperature and per-view bias
            logits = (x * cam_embed).sum(dim=-1) / (C ** 0.5) # (B, V)
        else:
            # no cam embedding: fall back to a learned per-view bias only
            logits = th.zeros(B, V, device=x.device)

        # add learned per-view bias 
        bias = self.view_bias[:V].unsqueeze(0)            # (1, V)
        logits = (logits + bias) / max(self.temperature, 1e-6)

        w = F.softmax(logits, dim=1)                      # (B, V)
        w = w.unsqueeze(-1)                               # (B, V, 1)

        # weighted sum over views
        g = (w * x).sum(dim=1)                            # (B, C_in)

        g = self.proj(g)                                  # (B, C_emb)
        # if self.ln_after is not None:
        #     g = self.ln_after(g)
        return g


class DINOCrossAttention(nn.Module):
    """
    한 번의 MHA에서 Self-Attn(BEV↔BEV) + Cross-Attn(BEV↔DINO)을 동시에 수행.

    Q = BEV
    K/V = concat([BEV(Self K/V with pos), DINO(Cross K/V with pos)], seq-dim)

    last_tokens: (B, V, Hp*Wp, C_dino)
    patch_hw:    (Hp, Wp)
    img_metas[b]: {'lidar2img': (V,4,4), 'img_shape': (H_img, W_img)}
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,  
        dino_channels=768,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        pos_scale_self=1.0,     
        pos_scale_dino=1.0,    
        use_ffn=False,         # optional FFN
        z_bins=8,               # Z
        num_points_in_pillar=4  # S (한 BEV 셀당 z 샘플 수)
        ):
        super().__init__()
        
        self.channels = channels
        if num_head_channels == -1:
            assert channels % num_heads == 0, \
                f"q,k,v channels {channels} is not divisible by num_heads {num_heads}"
            self.num_heads = num_heads
            self.d_k = channels // num_heads
        else:
            assert channels % num_head_channels == 0, \
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
            self.d_k = num_head_channels

        self.dino_channels = dino_channels
        self.pc_range = pc_range
        self.z_bins = int(z_bins)
        self.num_points_in_pillar = int(num_points_in_pillar)

        # ---- Q from BEV ----
        self.norm_q = normalization(self.channels)
        self.q_proj = conv_nd(1, self.channels, self.num_heads * self.d_k, kernel_size=1)

        # ---- Self K/V from BEV (content + BEV pos) ----
        self.self_pos_raw_dim = 2  # (x_norm, y_norm)
        self.self_pos_dim = max(32, int(self.channels * pos_scale_self))
        self.self_pos_mlp = nn.Sequential(
                                conv_nd(1, self.self_pos_raw_dim, self.self_pos_dim, 1),
                                nn.GELU(),
                                conv_nd(1, self.self_pos_dim, self.self_pos_dim, 1),
                            )
        # concat([bev_content(C), bev_pos(self_pos_dim)]) -> project to heads*d_k
        self.W_k_self = conv_nd(1, self.channels + self.self_pos_dim, self.num_heads * self.d_k, 1)
        self.W_v_self = conv_nd(1, self.channels, self.num_heads * self.d_k, 1)

        # ---- Cross K/V from DINO (content + DINO pos) ----
        self.dino_pos_raw_dim = 4  # (u_norm, v_norm, view_ratio, depth_norm)
        self.dino_pos_dim = max(32, int(self.channels * pos_scale_dino))
        self.dino_pos_mlp = nn.Sequential(
                                conv_nd(1, self.dino_pos_raw_dim, self.dino_pos_dim, 1),
                                nn.GELU(),
                                conv_nd(1, self.dino_pos_dim, self.dino_pos_dim, 1),
                            )
        # DINO content -> project to C (content embedding), then to heads*d_k
        self.dino_content_to_c = conv_nd(1, self.dino_channels, self.channels, 1)
        self.W_k_dino = conv_nd(1, self.channels + self.dino_pos_dim, self.num_heads * self.d_k, 1)
        self.W_v_dino = conv_nd(1, self.channels, self.num_heads * self.d_k, 1)

        # ---- Output projection (zero-init) ----
        self.proj_out = zero_module(conv_nd(1, self.num_heads * self.d_k, self.channels, 1))

        # ---- Optional FFN ----
        if use_ffn:
            self.ffn = nn.Sequential(
                            conv_nd(1, self.channels, 4*self.channels, 1),
                            nn.GELU(),
                            conv_nd(1, 4*self.channels, self.channels, 1),
                        )
        else:
            self.ffn = None

        # ---- Learnable gate for DINO branch (stability) ----
        self.dino_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

    @torch.no_grad()
    def _sample_dino_tokens(
        self,
        last_tokens,      # (B,V,Hp*Wp,Cd)
        patch_hw,
        uv,               # (B,V,N,2) pixel
        mask,             # (B,V,N) bool
        img_hw,
    ):
        """ bilinear sampling on patch grid → (B,V,N,Cd) """
        B, V, P, Cd = last_tokens.shape
        Hp, Wp = patch_hw
        # (B,V,Hp*Wp,Cd) -> (B,V,Cd,Hp,Wp)
        feat = last_tokens.view(B, V, Hp, Wp, Cd).permute(0,1,4,2,3).contiguous()
        grid = uv_to_patch_grid(uv, img_hw, patch_hw)                                # (B*V, N, 1, 2)
        feat_bv = feat.view(B*V, Cd, Hp, Wp)
        sampled = F.grid_sample(feat_bv, grid, mode="bilinear",
                                align_corners=False, padding_mode="zeros")            # (B*V,Cd,N,1)
        sampled = sampled.squeeze(-1).permute(0,2,1).contiguous()                     # (B*V,N,Cd)
        return sampled.view(B, V, -1, Cd)                                             # (B,V,N,Cd)

    def _make_z_samples(self, device, dtype):
        z_min, z_max = self.pc_range[2], self.pc_range[5]
        # [0,1] 정규화된 bin-center 샘플
        zs_norm = torch.linspace(0.5, self.z_bins - 0.5, steps=self.num_points_in_pillar,device=device, dtype=dtype) / float(self.z_bins)    # (S,)
        z_samples = z_min + zs_norm * (z_max - z_min)           
        return z_samples
    
    
    def forward(self, bev_feat, dino_cond):
        '''
        :param x: (N, C, H, W)
        :param dino_cond: {'feature_type', 'features', 'patch_hw', 'last_cls', 'last_tokens', 'img_metas', 'geom'}
        '''
        last_tokens = dino_cond['last_tokens']
        patch_hw = dino_cond['patch_hw']
        img_metas = dino_cond['img_metas']
        
        B, C, H, W = bev_feat.shape
        assert C == self.channels, "bev_feat channels must equal `channels` passed to the module."
        assert last_tokens.shape[-1] == self.dino_channels, \
            "last_tokens feature dim must equal `dino_channels`."
    
        device = bev_feat.device
        L1 = H * W
        
        z_samples = self._make_z_samples(device=device, dtype=bev_feat.dtype)  # (S,)
        S = z_samples.numel()
        
        Vcams = last_tokens.shape[1]
        Cd = last_tokens.shape[-1]
        Hp, Wp = patch_hw

        # ----- BEV grid & normalized coords (for self pos) -----
        bx, by = build_bev_xy_grid(B, H, W, self.pc_range, device)  # (B,H,W)
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        x_norm = ((bx - x_min) / max(1e-6, (x_max - x_min))).clamp(0,1).view(B, 1, L1)  # (B,1,L1)
        y_norm = ((by - y_min) / max(1e-6, (y_max - y_min))).clamp(0,1).view(B, 1, L1)  # (B,1,L1)

        # ----- Q from BEV -----
        x = bev_feat.view(B, C, L1)                         # (B,C,L1)
        Q = self.q_proj(self.norm_q(x))                        # (B, H*d_k, L1)

        # ----- Self K/V from BEV -----
        self_pos_raw = torch.cat([x_norm, y_norm], dim=1)         # (B,2,L1)
        self_pos = self.self_pos_mlp(self_pos_raw)                # (B,self_pos_dim,L1)
        K_self_in = torch.cat([x, self_pos], dim=1)               # (B,C+self_pos_dim,L1)
        K_self = self.W_k_self(K_self_in)                         # (B, H*d_k, L1)
        V_self = self.W_v_self(x)                                 # (B, H*d_k, L1)

        # ----- Cross DINO: align & aggregate to one token per BEV cell -----
        # 3D points: (B,V,N=L1*S, 3)
        bx1 = bx.view(B, L1); by1 = by.view(B, L1)
        bz = z_samples.view(1,1,S).expand(B, L1, S).contiguous()
        N = L1 * S
        pts = torch.stack([bx1.unsqueeze(-1).expand(-1,-1,S),
                           by1.unsqueeze(-1).expand(-1,-1,S),
                           bz], dim=-1).view(B, 1, N, 3).expand(-1, Vcams, -1, -1).contiguous()

        lidar2img = torch.stack(
            [torch.as_tensor(m['lidar2img'], device=device, dtype=bev_feat.dtype) for m in img_metas],
            dim=0
        )  # (B,V,4,4)
        H_img, W_img = img_metas[0]['img_shape'][0][:2]
        img_hw = (H_img, W_img)
        uv, mask, depth = project_lidar2img(pts, lidar2img, img_hw)                # (B,V,N,2),(B,V,N),(B,V,N)
        sampled = self._sample_dino_tokens(last_tokens, patch_hw, uv, mask, img_hw) # (B,V,N,Cd)

        # (view,z) 가중 평균 → (B,L1,Cd)
        sampled = sampled.view(B, Vcams, L1, S, Cd)
        mask_vz = mask.view(B, Vcams, L1, S)
        dino_mean_vs = masked_mean(sampled, mask_vz.unsqueeze(-1), dim=1)                # (B,L1,S,Cd)
        dino_mean    = masked_mean(dino_mean_vs, mask_vz.any(dim=1).unsqueeze(-1), dim=2) # (B,L1,Cd)

        # DINO pos raw (u_norm, v_norm, view_ratio, depth_norm) per BEV cell
        uv_ = uv.view(B, Vcams, L1, S, 2)
        uv_mean_vs = masked_mean(uv_, mask_vz.unsqueeze(-1), dim=1)                      # (B,L1,S,2)
        uv_mean    = masked_mean(uv_mean_vs, mask_vz.any(dim=1).unsqueeze(-1), dim=2)    # (B,L1,2)

        visible_views = mask_vz.any(dim=3).float().sum(dim=1)                              # (B,L1)
        view_ratio   = (visible_views / float(Vcams)).clamp(0, 1).view(B, 1, L1)

        u_norm = (uv_mean[..., 0] / max(1, W_img - 1)).clamp(0,1).view(B, 1, L1)
        v_norm = (uv_mean[..., 1] / max(1, H_img - 1)).clamp(0,1).view(B, 1, L1)
        depth_mean = masked_mean(depth.view(B, Vcams, L1, S), mask_vz, dim=1)
        depth_mean = masked_mean(depth_mean, mask_vz.any(dim=1), dim=2).view(B, 1, L1)
        z_min = max(1e-6, self.pc_range[2]); z_max = max(z_min + 1e-3, self.pc_range[5])
        d_norm = ((depth_mean - z_min) / (z_max - z_min)).clamp(0,1)                      # (B,1,L1)

        # DINO content/pos -> heads*d_k
        dino_c = self.dino_content_to_c(dino_mean.transpose(1,2).contiguous())            # (B,C,L1)
        dino_pos_raw = torch.cat([u_norm, v_norm, view_ratio, d_norm], dim=1)             # (B,4,L1)
        dino_pos = self.dino_pos_mlp(dino_pos_raw)                                        # (B,dino_pos_dim,L1)

        K_dino = self.W_k_dino(torch.cat([dino_c, dino_pos], dim=1))                      # (B,H*d_k,L1)
        V_dino = self.W_v_dino(dino_c)                                                    # (B,H*d_k,L1)

        # ---- Mixed memory: concat on sequence dim ----
        g = torch.sigmoid(self.dino_gate)                                                 # scalar in (0,1)
        K_mix = torch.cat([K_self, K_dino], dim=2)                                        # (B,H*d_k, 2*L1)
        V_mix = torch.cat([V_self, g * V_dino], dim=2)                                    # (B,H*d_k, 2*L1)

        # ---- Multi-Head Attention (per-head) ----
        B_, Hd = B, self.num_heads * self.d_k
        Lq, Lk = L1, K_mix.shape[2]
        Qh = Q.view(B_, self.num_heads, self.d_k, Lq).reshape(B_*self.num_heads, self.d_k, Lq)
        Kh = K_mix.view(B_, self.num_heads, self.d_k, Lk).reshape(B_*self.num_heads, self.d_k, Lk)
        Vh = V_mix.view(B_, self.num_heads, self.d_k, Lk).reshape(B_*self.num_heads, self.d_k, Lk)

        # FP16-friendly scaling
        scale = 1.0 / math.sqrt(math.sqrt(float(self.d_k)))
        attn_logits = torch.einsum("bcl, bcs -> bls", Qh*scale, Kh*scale)                 # (B*H, Lq, Lk)
        attn = F.softmax(attn_logits.float(), dim=-1).type_as(attn_logits)
        out_h = torch.einsum("bls, bcs -> bcl", attn, Vh)                                 # (B*H, d_k, Lq)
        out = out_h.view(B_, self.num_heads, self.d_k, Lq).reshape(B_, Hd, Lq)            # (B,H*d_k,L1)

        out = self.proj_out(out)                                                          # (B,C,L1)
        y = (x + out).view(B, C, H, W)                                                    # residual add

        if self.ffn is not None:
            y_ffn = self.ffn(y.view(B, C, L1)).view(B, C, H, W)
            y = y + y_ffn

        return y


class MultiScaleConcat(nn.Module):
    def __init__(self, in_chs=[256,512,1024,1024], out_dim=256, mid=256):
        """
        Hierarchical feature fusion by simple concatenation and conv
        """
        super().__init__()

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(in_chs), 2*mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, 2*mid),           
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, 2*mid, kernel_size=3, padding=1, groups=2*mid, bias=False),
            nn.GroupNorm(32, 2*mid),
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_dim),       
        )

    def forward(self, xs):
        x0, x1, x2, x3 = xs
        B, _, H, W = x0.shape
        B, _, H1, W1 = x1.shape
                    
        f1 = th.cat([x2, x3], dim=1)     # H//4, W//4
        f2 = th.cat([x1, F.interpolate(f1, (H1, W1), mode='bilinear', align_corners=False)], dim=1)     # H//2, W//2
        x = th.cat([x0, F.interpolate(f2, (H, W), mode='bilinear', align_corners=False)], dim=1)     # H, W  
        return self.out_layer(x)           # [B,out_dim,H,W]
    
    
class MultiScaleConcatV2(nn.Module):
    def __init__(self, in_chs=[256,512,1024], out_dim=256, mid=256):
        """
        Hierarchical feature fusion by simple concatenation and conv 
        (without mid-block)
        """
        super().__init__()

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(in_chs), 2*mid, kernel_size=1, bias=False),
            nn.GroupNorm(32, 2*mid),           
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, 2*mid, kernel_size=3, padding=1, groups=2*mid, bias=False),
            nn.GroupNorm(32, 2*mid),
            nn.SiLU(inplace=True),

            nn.Conv2d(2*mid, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_dim),       
        )

    def forward(self, xs):
        x0, x1, x2 = xs
        B, _, H, W = x0.shape
        B, _, H1, W1 = x1.shape
                    
        f1 = th.cat([x1, F.interpolate(x2, (H1, W1), mode='bilinear', align_corners=False)], dim=1)     # H//2, W//2
        f2 = th.cat([x0, F.interpolate(f1, (H, W), mode='bilinear', align_corners=False)], dim=1)     # H, W  
        return self.out_layer(f2)           # [B,out_dim,H,W]



class LayoutDiffusionUNetModel(nn.Module):
    """
    A UNetModel that conditions on layout with an encoding transformer.
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_ds: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.

    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param {
        layout_length: number of layout objects to expect.
        hidden_dim: width of the transformer.
        num_layers: depth of the transformer.
        num_heads: heads in the transformer.
        xf_final_ln: use a LayerNorm after the output layer.
        num_classes_for_layout_object: num of classes for layout object.
        mask_size_for_layout_object: mask size for layout object image.
    }

    """

    def __init__(
            self,
            layout_encoder,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_ds,
            encoder_channels=None,
            dino_dim=768,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_positional_embedding_for_attention=False,
            use_spatial_transformer=False,
            image_size=256,
            attention_block_type='GLIDE',
            num_attention_blocks=1,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False,
            num_pre_downsample=0,
            transformer_depth=1,
            return_multiscale=True,
            multiscale_indices='auto',
            legacy=True,
    ):
        super().__init__()

        self.norm_for_obj_embedding = norm_for_obj_embedding
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.norm_first = norm_first
        self.use_key_padding_mask=use_key_padding_mask
        self.num_attention_blocks = num_attention_blocks
        self.attention_block_type = attention_block_type
        if self.attention_block_type == 'GLIDE':
            attention_block_fn = AttentionBlock
        elif self.attention_block_type == 'ObjectAwareCrossAttention':
            attention_block_fn = ObjectAwareCrossAttention

        self.image_size = image_size
        self.use_positional_embedding_for_attention = use_positional_embedding_for_attention

        self.layout_encoder = layout_encoder

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_ds = attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # multi-scale features index
        self.return_multiscale = return_multiscale

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.downsample_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        for _ in range(num_pre_downsample):
            self.downsample_blocks.append(Downsample(
                            in_channels, conv_resample, dims=dims, out_channels=in_channels
                        ))
            self.upsample_blocks.append(Upsample(
                            out_channels, conv_resample, dims=dims, out_channels=out_channels
                        ))
            self.image_size = self.image_size // 2  

        # DINO feature condition
        # self.adapter = DINOContextAdapter(c_in=dino_dim, c_emb=time_embed_dim, pool='mean')
        # self.adapter = DINOContextAdapter(c_in=dino_dim, c_emb=1024, num_views=6)

        self.multi_concat = MultiScaleConcat(in_chs=(model_channels, model_channels*2, model_channels*4, model_channels*4), 
                                                    out_dim=out_channels, 
                                                    mid=model_channels)
        # self.multi_concat = MultiScaleConcatV2(in_chs=(model_channels, model_channels*2, model_channels*4), 
        #                                        out_dim=out_channels, 
        #                                        mid=model_channels)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    print('encoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    for _ in range(self.num_attention_blocks):
                        if ds in [1, 2]:
                            layers.append(
                                attention_block_fn(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                                    encoder_channels=encoder_channels,
                                    ds=ds,
                                    resolution=int(self.image_size // ds),
                                    type='input',
                                    use_positional_embedding=self.use_positional_embedding_for_attention,
                                    use_key_padding_mask=self.use_key_padding_mask,
                                    channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                    norm_first=self.norm_first,
                                    norm_for_obj_embedding=self.norm_for_obj_embedding
                                )
                            )
                        elif ds == 4:
                            layers.append(
                                DINOCrossAttention(
                                    ch, 
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                                    dino_channels=dino_dim,
                                    pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                                    pos_scale_self=1.0,     
                                    pos_scale_dino=1.0,
                                    use_ffn=False,
                                    z_bins=8,              
                                    num_points_in_pillar=4  
                                )
                            )
                # self.input_blocks.append(TimestepEmbedSequential(*layers))
                block = TimestepEmbedSequential(*layers)
                block.ctx_ds = ds    
                self.input_blocks.append(block)
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                block = TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                            if resblock_updown
                            else Downsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                block.ctx_ds = ds  
                self.input_blocks.append(block)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        print('middle attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            DINOCrossAttention(
                ch, 
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                dino_channels=dino_dim,
                pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                pos_scale_self=1.0,     
                pos_scale_dino=1.0,
                use_ffn=False,
                z_bins=8,              
                num_points_in_pillar=4  
            ),
            # ResBlock(
            #     ch,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
            attention_block_fn(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                ds=ds,
                resolution=int(self.image_size // ds),
                type='middle',
                use_positional_embedding=self.use_positional_embedding_for_attention,
                use_key_padding_mask=self.use_key_padding_mask,
                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                norm_first=self.norm_first,
                norm_for_obj_embedding=self.norm_for_obj_embedding
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block.ctx_ds = ds
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    print('decoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    for _ in range(self.num_attention_blocks):
                        if ds in [1, 2]:
                            layers.append(
                                attention_block_fn(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=num_head_channels,
                                    encoder_channels=encoder_channels,
                                    ds=ds,
                                    resolution=int(self.image_size // ds),
                                    type='output',
                                    use_positional_embedding=self.use_positional_embedding_for_attention,
                                    use_key_padding_mask=self.use_key_padding_mask,
                                    channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                    norm_first=self.norm_first,
                                    norm_for_obj_embedding=self.norm_for_obj_embedding
                                )
                            )
                        elif ds == 4:
                            layers.append(
                                DINOCrossAttention(
                                    ch, 
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                                    dino_channels=dino_dim,
                                    pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                                    pos_scale_self=1.0,     
                                    pos_scale_dino=1.0,
                                    use_ffn=False,
                                    z_bins=8,              
                                    num_points_in_pillar=4  
                                )
                            )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            out_size=int(self.image_size // (ds // 2))
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, out_size=int(self.image_size // ds))
                    )
                    ds //= 2
                # self.output_blocks.append(TimestepEmbedSequential(*layers))
                block = TimestepEmbedSequential(*layers)
                block.ctx_ds = ds 
                self.output_blocks.append(block)
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.layout_encoder.convert_to_fp16()

    def forward(self, x, timesteps, dino_cond, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, obj_name=None, **kwargs):
        hs, extra_outputs = [], []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        layout_outputs = self.layout_encoder(
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj,
            obj_name=obj_name
        )
        xf_proj, xf_out = layout_outputs["xf_proj"], layout_outputs["xf_out"]  # xf_proj: (B, 1024), xf_out: (B, 256, 300)
        
        # B, V, _ = dino_cond['last_cls'].shape
        # cam_ids = th.arange(V, dtype=th.long, device=emb.device).unsqueeze(0).expand(B, V)
        # dino_cond_proj = self.adapter(dino_cond['last_cls'], cam_ids=cam_ids)

        emb = emb + xf_proj.to(emb) # emb: (B, 1024)
        
        # emb = emb + xf_proj.to(emb)+ dino_cond_proj.to(emb)  # emb: (B, 1024)
        
        out_list = []
        h = x.type(self.dtype)  # h: (B, C, H, W)
        for module in self.downsample_blocks:
            h = module(h) 

        for module in self.input_blocks:
            h, extra_output = module(h, emb, dino_cond, layout_outputs) 
            # h, extra_output = module(h, emb, layout_outputs) 
            if extra_output is not None:
                extra_outputs.append(extra_output)
            hs.append(h)

        h, extra_output = self.middle_block(h, emb, dino_cond, layout_outputs) 
        # h, extra_output = self.middle_block(h, emb, layout_outputs)
        if extra_output is not None:
            extra_outputs.append(extra_output)
        out_list = []
        out_list.append(h)

        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h, extra_output = module(h, emb, dino_cond, layout_outputs) 
            # h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
            if self.return_multiscale and i_out in [1, 4]:
                out_list.append(h)
            
        h = h.type(x.dtype)
        h = self.out(h)
        out_list.append(h)
        for module in self.upsample_blocks:
            h = module(h)

        if self.return_multiscale:
            multi_feat = self.multi_concat(out_list[::-1]) 
            return h, multi_feat, out_list
        else:
            return [h, extra_outputs]
    
    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        weights_name = SAFETENSORS_WEIGHTS_NAME
        safetensors.torch.save_file(self.state_dict(), os.path.join(save_directory, weights_name), metadata={"format": "pt"})
        
    def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
        weights_name = SAFETENSORS_WEIGHTS_NAME
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, weights_name)
            elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        else:
            print(f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.")
            return
        state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
        try:
            self.load_state_dict(state_dict, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')
            self.load_state_dict(state_dict, strict=False)

