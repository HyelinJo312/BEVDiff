"""Checks for the encoder-only SBAM placement in the v5_4 UNet."""

import unittest

import torch

from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5_4 import (
    DiffusionUNetModel,
    SBAMResBlock,
    SelfAttnTransformer,
)


class EncoderSBAMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def make_model(attention_ds):
        return DiffusionUNetModel(
            seg_bev_aligner=dict(bev_h=50, bev_w=50, emb_channels=32,
                                 channel_mult=[1, 1, 1]),
            in_channels=32, model_channels=32, seg_channels=[32, 32, 32],
            out_channels=32, num_res_blocks=2, attention_ds=attention_ds,
            channel_mult=[1, 2, 4], image_size=50, resblock_updown=True,
            use_global_seg=False, use_scale_shift_norm=True,
            sbam_use_scale_shift_norm=False,
        )

    def test_six_encoder_sbam_blocks_and_unchanged_attention_counts(self):
        model = self.make_model([1, 2, 4])
        encoder_sbam = [
            (block.ctx_ds, layer)
            for block in model.input_blocks
            for layer in block
            if isinstance(layer, SBAMResBlock)
        ]
        self.assertEqual([ds for ds, _ in encoder_sbam], [1, 1, 2, 2, 4, 4])
        self.assertTrue(all(not block.use_scale_shift_norm
                            for _, block in encoder_sbam))
        self.assertFalse(any(isinstance(layer, SBAMResBlock)
                             for block in model.output_blocks for layer in block))
        self.assertFalse(any(isinstance(layer, SBAMResBlock)
                             for layer in model.middle_block))
        self.assertEqual(sum(isinstance(layer, SelfAttnTransformer)
                             for block in model.input_blocks for layer in block), 6)
        self.assertEqual(sum(isinstance(layer, SelfAttnTransformer)
                             for layer in model.middle_block), 1)
        self.assertEqual(sum(isinstance(layer, SelfAttnTransformer)
                             for block in model.output_blocks for layer in block), 9)

    def test_forward_and_condition_gradients_at_all_scales(self):
        model = self.make_model([])
        with torch.no_grad():
            model.out[-1].weight.normal_(std=0.01)
            for block in model.input_blocks:
                for layer in block:
                    if isinstance(layer, SBAMResBlock):
                        layer.out_layers[-1].weight.normal_(std=0.01)

        x = torch.randn(1, 32, 50, 50)
        seg_bev_maps = {
            ds: torch.randn(1, 32, size, size, requires_grad=True)
            for ds, size in [(1, 50), (2, 25), (4, 12)]
        }
        prediction = model(x, torch.tensor([100]), img_metas=None,
                           seg_bev_maps=seg_bev_maps)
        self.assertEqual(prediction.shape, x.shape)
        prediction.square().mean().backward()
        for feature in seg_bev_maps.values():
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
