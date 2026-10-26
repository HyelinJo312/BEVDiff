"""Focused checks for learned-query global semantic pooling in v5_5."""

import unittest

import torch

from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import (
    GlobalSemEncoder,
    semantic_input,
)
from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5_5 import (
    DiffusionUNetModel,
    SBAMResBlock,
    ResBlock,
)


class GlobalSemEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_variable_spatial_shape_and_attention_gradients(self):
        encoder = GlobalSemEncoder(32)
        self.assertGreater(encoder.query.abs().sum().item(), 0)
        for height, width in [(5, 7), (12, 12)]:
            feature = torch.randn(2, 32, height, width, requires_grad=True)
            output = encoder(feature)
            self.assertEqual(output.shape, (2, 32))
            output.square().mean().backward()
            for grad in (feature.grad, encoder.query.grad,
                         encoder.attention.in_proj_weight.grad):
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
            key_grad, value_grad = encoder.attention.in_proj_weight.grad[32:].chunk(2)
            self.assertGreater(key_grad.abs().sum().item(), 0)
            self.assertGreater(value_grad.abs().sum().item(), 0)
            encoder.zero_grad(set_to_none=True)

        restored = GlobalSemEncoder(32)
        restored.load_state_dict(encoder.state_dict(), strict=True)
        self.assertIn('query', encoder.state_dict())
        torch.testing.assert_close(restored(feature.detach()),
                                   encoder(feature.detach()))

    @staticmethod
    def make_model(semantic_type):
        return DiffusionUNetModel(
            seg_bev_aligner=dict(bev_h=50, bev_w=50, num_classes=16,
                                 emb_channels=32, channel_mult=[1, 1, 1],
                                 semantic_type=semantic_type),
            in_channels=32, model_channels=32, seg_channels=[32, 32, 32],
            out_channels=32, num_res_blocks=2, attention_ds=[],
            channel_mult=[1, 2, 4], image_size=50, resblock_updown=True,
            use_scale_shift_norm=True, sbam_use_scale_shift_norm=False,
        )

    def test_unet_cached_and_encoded_paths_with_global_gradient(self):
        share = torch.rand(1, 17, 50, 50)
        counts = torch.rand_like(share) * 100
        x = torch.randn(1, 32, 50, 50)
        timestep = torch.tensor([100])
        for semantic_type in ('share', 'share+logcount'):
            with self.subTest(semantic_type=semantic_type):
                model = self.make_model(semantic_type)
                self.assertIsInstance(model.seg_global_embed[0], GlobalSemEncoder)
                self.assertIsInstance(model.middle_block[0], SBAMResBlock)
                self.assertIsInstance(model.middle_block[2], ResBlock)
                self.assertEqual(model.middle_block.ctx_ds, 4)
                self.assertFalse(model.middle_block[0].use_scale_shift_norm)
                self.assertEqual(sum(isinstance(layer, SBAMResBlock)
                                     for block in model.input_blocks for layer in block), 6)
                self.assertFalse(any(isinstance(layer, SBAMResBlock)
                                     for block in model.output_blocks for layer in block))
                self.assertEqual(torch.count_nonzero(model.seg_global_embed[-1].weight).item(), 0)
                self.assertEqual(torch.count_nonzero(model.seg_global_embed[-1].bias).item(), 0)

                semantic_prob = semantic_input(share, counts, semantic_type)
                encoded = model.encode_seg(None, None, semantic_prob=semantic_prob)
                self.assertEqual(torch.count_nonzero(
                    model.seg_global_embed(encoded[1])).item(), 0)

                # Wake zero-initialized projections for a one-step gradient check.
                with torch.no_grad():
                    model.out[-1].weight.normal_(std=0.01)
                    model.seg_global_embed[-1].weight.normal_(std=0.01)
                    for block in [*model.input_blocks, model.middle_block]:
                        for layer in block:
                            if isinstance(layer, SBAMResBlock):
                                layer.out_layers[-1].weight.normal_(std=0.01)

                for cached in (semantic_prob, torch.zeros_like(semantic_prob)):
                    encoded = model.encode_seg(None, None, semantic_prob=cached)
                    output_cached = model(x, timestep, None, semantic_prob=cached)
                    output_encoded = model(x, timestep, None, seg_bev_maps=encoded)
                    self.assertEqual(output_cached.shape, x.shape)
                    torch.testing.assert_close(output_cached, output_encoded)

                # Use observed semantics to check spatial modulation weights.
                model(x, timestep, None, semantic_prob=semantic_prob).square().mean().backward()
                for norm in (model.middle_block[0].norm_0, model.middle_block[0].norm_1):
                    for parameter in (norm.conv_gamma.weight, norm.conv_beta.weight):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                query_grad = model.seg_global_embed[0].query.grad
                self.assertIsNotNone(query_grad)
                self.assertTrue(torch.isfinite(query_grad).all())
                self.assertGreater(query_grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
