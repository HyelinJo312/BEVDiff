"""Focused checks for the optional SBAMResBlock timestep scale/shift path."""

import unittest
from pathlib import Path

import torch
from mmcv import Config

from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5 import (
    DiffusionUNetModel,
    SBAMResBlock,
)


class SBAMScaleShiftNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 32, 8, 8)
        self.emb = torch.randn(2, 128)
        self.seg = torch.randn(2, 16, 8, 8)

    @staticmethod
    def _wake_residual(block):
        with torch.no_grad():
            block.out_layers[-1].weight.normal_(std=0.01)

    def test_default_branch_keeps_legacy_shape_state_and_forward(self):
        default = SBAMResBlock(32, 128, 0, out_channels=64, seg_channels=16)
        explicit = SBAMResBlock(
            32, 128, 0, out_channels=64, seg_channels=16,
            use_scale_shift_norm=False,
        )
        explicit.load_state_dict(default.state_dict(), strict=True)
        self.assertEqual(default.emb_layers[-1].weight.shape, (64, 128))
        self._wake_residual(default)

        h = default.in_layers(default.norm_0(self.x, self.seg, self.emb))
        emb_out = default.emb_layers(self.emb).to(h.dtype)[:, :, None, None]
        expected = default.skip_connection(self.x) + default.out_layers(
            default.norm_1(h + emb_out, self.seg, self.emb)
        )
        torch.testing.assert_close(default(self.x, self.emb, self.seg), expected,
                                   rtol=0, atol=0)

    def test_scale_shift_matches_formula_and_backpropagates(self):
        block = SBAMResBlock(
            32, 128, 0, out_channels=64, seg_channels=16,
            use_scale_shift_norm=True,
        )
        self.assertEqual(block.emb_layers[-1].weight.shape, (128, 128))
        self._wake_residual(block)
        x = self.x.clone().requires_grad_()
        emb = self.emb.clone().requires_grad_()
        seg = self.seg.clone().requires_grad_()

        h = block.in_layers(block.norm_0(x, seg, emb))
        scale, shift = block.emb_layers(emb).to(h.dtype).chunk(2, dim=1)
        expected = block.skip_connection(x) + block.out_layers(
            block.norm_1(h, seg, emb) * (1 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )
        actual = block(x, emb, seg)
        self.assertEqual(actual.shape, (2, 64, 8, 8))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        for grad in (x.grad, emb.grad, seg.grad,
                     block.emb_layers[-1].weight.grad,
                     block.norm_1.conv_gamma.weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_unet_flag_only_reaches_decoder_sbam_blocks(self):
        settings = dict(
            seg_bev_aligner=dict(bev_h=8, bev_w=8, emb_channels=32,
                                 channel_mult=[1, 1, 1]),
            in_channels=32, model_channels=32, seg_channels=[32, 32, 32],
            out_channels=32, num_res_blocks=1, attention_ds=[],
            channel_mult=[1, 1, 1], use_global_seg=False,
            use_scale_shift_norm=True, sbam_use_scale_shift_norm=True,
        )
        model = DiffusionUNetModel(**settings)
        sbam_blocks = [module for module in model.modules()
                       if isinstance(module, SBAMResBlock)]
        self.assertTrue(sbam_blocks)
        self.assertTrue(all(block.use_scale_shift_norm for block in sbam_blocks))
        self.assertFalse(any(isinstance(module, SBAMResBlock)
                             for block in model.input_blocks
                             for module in block.modules()))
        self.assertFalse(any(isinstance(module, SBAMResBlock)
                             for module in model.middle_block.modules()))

    def test_stage1_variant_sets_only_new_flag(self):
        config_dir = (Path(__file__).resolve().parents[1] / 'projects' /
                      'configs' / 'bevdiffuser')
        base = Config.fromfile(str(config_dir / 'bev_tiny_onlyseg_sam_v2_metric3d.py'))
        variant = Config.fromfile(str(config_dir /
                         'bev_tiny_onlyseg_sam_v2_metric3d_sbam_scale_shift.py'))
        base_params = dict(base.unet.parameters)
        variant_params = dict(variant.unet.parameters)
        self.assertNotIn('sbam_use_scale_shift_norm', base_params)
        self.assertTrue(variant_params.pop('sbam_use_scale_shift_norm'))
        self.assertEqual(variant_params, base_params)


if __name__ == '__main__':
    unittest.main()
