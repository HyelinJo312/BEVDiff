"""Focused CPU checks for the Stage 2 cell-wise partial-channel MGD mask."""

import ast
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import torch


BEVFORMER = Path(__file__).resolve().parents[1]
DETECTOR = BEVFORMER / 'projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer_onlyseg.py'
CONFIG = BEVFORMER / 'projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_metric3d.py'


def load_mask_function():
    # The detector's optional mmdet/mmdet3d imports are not needed to test the
    # actual pure-torch mask function.
    tree = ast.parse(DETECTOR.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'sample_spatialchannel_keep_mask')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DETECTOR), 'exec'), namespace)
    return namespace[node.name]


class SpatialChannelMGDMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_mask = staticmethod(load_mask_function())

    def test_selected_cells_drop_independent_channels_only(self):
        x = torch.ones((1, 3, 2, 2), dtype=torch.float64, requires_grad=True)
        cells = torch.tensor([[[[0.1, 0.9], [0.9, 0.2]]]])
        channels = torch.tensor([[[[0.1, 0.9], [0.9, 0.9]],
                                  [[0.9, 0.9], [0.9, 0.1]],
                                  [[0.9, 0.9], [0.9, 0.9]]]])
        with patch.object(torch, 'rand', side_effect=[cells, channels]) as draw:
            mask = self.sample_mask(x, 0.5, 0.5)
        self.assertEqual(draw.call_args_list[0].args[0], (1, 1, 2, 2))
        self.assertEqual(draw.call_args_list[1].args[0], (1, 3, 2, 2))
        self.assertEqual(mask.shape, x.shape)
        self.assertEqual(mask.dtype, x.dtype)
        self.assertEqual(mask.device, x.device)
        expected = torch.ones_like(x)
        expected[0, 0, 0, 0] = 0
        expected[0, 1, 1, 1] = 0
        torch.testing.assert_close(mask, expected)
        (x * mask).sum().backward()
        torch.testing.assert_close(x.grad, expected)

    def test_expected_drop_rate_and_ratio_validation(self):
        torch.manual_seed(7)
        x = torch.ones((2, 32, 80, 80))
        mask = self.sample_mask(x, 0.8, 0.6875)
        self.assertAlmostEqual((mask == 0).float().mean().item(), 0.55, delta=0.01)
        for cell_ratio, channel_ratio in ((-0.1, 0.5), (0.8, 1.1)):
            with self.assertRaises(ValueError):
                self.sample_mask(x, cell_ratio, channel_ratio)
        with self.assertRaises(ValueError):
            self.sample_mask(x[0], 0.8, 0.6875)

    def test_config_and_default_spatial_branch_are_wired(self):
        model = runpy.run_path(str(CONFIG))['model']
        self.assertTrue(model['use_mgd'])
        self.assertTrue(model['use_spatialchannel'])
        self.assertEqual(model['spatialchannel_cell_ratio'] *
                         model['spatialchannel_channel_ratio'], model['mgd_lambda'])

        source = DETECTOR.read_text()
        tree = ast.parse(source)
        detector = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                        and n.name == 'DiffBEVFormerSegV3')
        constructor = next(n for n in detector.body if isinstance(n, ast.FunctionDef)
                           and n.name == '__init__')
        defaults = dict(zip((a.arg for a in constructor.args.kwonlyargs),
                            constructor.args.kw_defaults))
        self.assertFalse(ast.literal_eval(defaults['use_spatialchannel']))
        mask_branch = next(n for n in ast.walk(detector)
                           if isinstance(n, ast.If) and
                           isinstance(n.test, ast.Attribute) and
                           n.test.attr == 'use_spatialchannel')
        fallback = '\n'.join(ast.get_source_segment(source, n) for n in mask_branch.orelse)
        self.assertIn('torch.rand((B, 1, H, W)', fallback)
        self.assertIn('self.mgd_lambda', fallback)


if __name__ == '__main__':
    unittest.main()
