"""Focused CPU checks for the standalone v6 semantic BEV encoder."""

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = (Path(__file__).resolve().parents[1] / 'projects' / 'bevdiffuser'
               / 'layout_diffusion' / 'seg_bev_aligner_one_hot_v6.py')
spec = importlib.util.spec_from_file_location('seg_bev_aligner_one_hot_v6', MODULE_PATH)
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)


class SegBEVAlignerV6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shape_and_gradient_for_current_occupancy_condition(self):
        model = v6.SegBEVAligner(
            bev_h=50, bev_w=50, num_classes=16,
            emb_channels=16, channel_mult=(1, 1, 1))
        probabilities = torch.rand(2, 17, 50, 50, requires_grad=True)
        maps = model.encode_prob(probabilities)
        self.assertEqual({scale: tuple(value.shape) for scale, value in maps.items()},
                         {1: (2, 16, 50, 50), 2: (2, 16, 25, 25),
                          4: (2, 16, 12, 12)})
        sum(value.square().mean() for value in maps.values()).backward()
        self.assertIsNotNone(probabilities.grad)
        self.assertGreater(probabilities.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_odd_rectangular_shapes_and_channel_expansion(self):
        encoder = v6.SegBEVEncoder(8, channel_mult=(1, 2, 4))
        maps = encoder(torch.randn(1, 8, 51, 49))
        self.assertEqual({scale: tuple(value.shape) for scale, value in maps.items()},
                         {1: (1, 8, 51, 49), 2: (1, 16, 26, 25),
                          4: (1, 32, 12, 12)})

    def test_encoder_uses_original_three_convolution_stages(self):
        encoder = v6.SegBEVEncoder(8, channel_mult=(1, 2, 4))
        for block, channels, stride, padding in (
                (encoder.down0, (8, 8), (1, 1), (1, 1)),
                (encoder.down1, (8, 16), (2, 2), (1, 1)),
                (encoder.down2, (16, 32), (2, 2), (0, 0))):
            self.assertIsInstance(block, nn.Sequential)
            self.assertEqual(len(block), 2)
            self.assertIsInstance(block[0], nn.Conv2d)
            self.assertIsInstance(block[1], nn.SiLU)
            self.assertEqual((block[0].in_channels, block[0].out_channels), channels)
            self.assertEqual(block[0].kernel_size, (3, 3))
            self.assertEqual(block[0].stride, stride)
            self.assertEqual(block[0].padding, padding)

    def test_first_embedding_is_pointwise(self):
        model = v6.SegBEVAligner(
            bev_h=9, bev_w=9, num_classes=3, emb_channels=8,
            channel_mult=(1, 1, 1))
        self.assertIsInstance(model.prob_to_emb, nn.Conv2d)
        self.assertEqual(model.prob_to_emb.kernel_size, (1, 1))
        a = torch.zeros(1, 4, 9, 9)
        b = a.clone()
        b[:, 2, 4, 4] = 1
        with torch.no_grad():
            difference = model.prob_to_emb(b) - model.prob_to_emb(a)
        self.assertEqual(torch.count_nonzero(difference[:, :, :4, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(difference[:, :, 5:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(difference[:, :, 4, :4]).item(), 0)
        self.assertEqual(torch.count_nonzero(difference[:, :, 4, 5:]).item(), 0)
        self.assertEqual(tuple(model.encode_prob(a)[4].shape), (1, 8, 2, 2))


if __name__ == '__main__':
    unittest.main()
