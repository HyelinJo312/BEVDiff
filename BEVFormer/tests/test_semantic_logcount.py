"""share+logcount semantic input: encoding, aligner, cache layout and manifest check."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import projects.mmdet3d_plugin  # noqa: F401  (registers datasets/modules)
from projects.bevdiffuser.data_utils import (
    CustomNuScenesDiffusionDataset_seg_depth_v2, configure_semantic_type,
)
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import (
    SegBEVAligner, semantic_input,
)
from projects.bevdiffuser.semantic_bev_cache import (
    SemanticBEVCache, cache_channels, cache_contract, split_cache_array,
)


def aligner(**overrides):
    settings = dict(bev_h=8, bev_w=8, pc_range=(-4, -4, -2, 4, 4, 2), num_classes=2,
                    emb_channels=8, channel_mult=(1, 1, 1), depth_range=(.1, 10),
                    splat_mode='nearest')
    settings.update(overrides)
    return SegBEVAligner(**settings)


def two_pixel_maps():
    """One camera: two class-1 pixels and one class-2 pixel landing in one cell."""
    seg = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    depth = torch.zeros(1, 1, 4, 4)
    seg[0, 0, 1, 1] = seg[0, 0, 1, 2] = 1
    seg[0, 0, 2, 1] = 2
    depth[0, 0, 1, 1] = depth[0, 0, 1, 2] = depth[0, 0, 2, 1] = 1
    meta = dict(lidar2img=[np.eye(4)], pad_shape=[(4, 4, 3)])
    return seg, depth, meta


class SemanticInputTest(unittest.TestCase):
    def test_share_is_identity_and_logcount_formula(self):
        share = np.random.rand(3, 4, 4).astype(np.float32)
        counts = np.random.rand(3, 4, 4).astype(np.float32) * 1e4
        self.assertIs(semantic_input(share, counts, 'share'), share)
        out = semantic_input(share, counts, 'share+logcount', 1e5)
        self.assertEqual(out.shape, (6, 4, 4))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out[:3], share)
        np.testing.assert_allclose(out[3:], np.log1p(counts) / math.log1p(1e5), rtol=1e-6)

    def test_numpy_and_torch_paths_match(self):
        share = np.random.rand(2, 3, 4, 4).astype(np.float32)
        counts = (np.random.rand(2, 3, 4, 4) * 500).astype(np.float32)
        a = semantic_input(share, counts, 'share+logcount')
        b = semantic_input(torch.from_numpy(share), torch.from_numpy(counts), 'share+logcount')
        np.testing.assert_allclose(a, b.numpy(), rtol=1e-6, atol=1e-7)

    def test_half_counts_use_float32_log_without_mutation(self):
        share = torch.ones(1, 3, 2, 2)
        counts = torch.tensor([0.125, 1.5, 300, 2000], dtype=torch.float16).reshape(1, 1, 2, 2).expand_as(share)
        original = counts.clone()
        result = semantic_input(share, counts, 'share+logcount')
        expected = torch.log1p(counts.float()) / math.log1p(1e5)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result[:, 3:], expected, rtol=0, atol=0)
        torch.testing.assert_close(counts, original, rtol=0, atol=0)
        numpy_result = semantic_input(share.numpy(), counts.numpy(), 'share+logcount')
        np.testing.assert_allclose(numpy_result, result.numpy(), rtol=1e-6, atol=1e-7)

    def test_invalid_type_or_missing_counts_rejected(self):
        share = np.zeros((3, 4, 4), np.float32)
        with self.assertRaisesRegex(ValueError, 'semantic_type'):
            semantic_input(share, share, 'presence')
        with self.assertRaisesRegex(ValueError, 'requires raw per-class counts'):
            semantic_input(share, None, 'share+logcount')


class AlignerTest(unittest.TestCase):
    def test_default_aligner_unchanged(self):
        legacy, explicit = aligner(), aligner(semantic_type='share')
        self.assertEqual(legacy.input_channels, 3)
        self.assertEqual({k: v.shape for k, v in legacy.state_dict().items()},
                         {k: v.shape for k, v in explicit.state_dict().items()})
        seg, depth, meta = two_pixel_maps()
        result = legacy.project_semantics(seg, [meta], depth)
        self.assertEqual(len(result), 2)

    def test_logcount_input_width_and_mismatch_error(self):
        model = aligner(semantic_type='share+logcount')
        self.assertEqual(model.prob_to_emb[0].in_channels, 6)
        self.assertEqual(model.encode_prob(torch.rand(2, 6, 8, 8))[1].shape, (2, 8, 8, 8))
        with self.assertRaisesRegex(ValueError, 'semantic_type must match'):
            model.encode_prob(torch.rand(2, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, 'semantic_type'):
            aligner(semantic_type='presence')

    def test_return_counts_consistent_with_share(self):
        seg, depth, meta = two_pixel_maps()
        # num_classes=3 so label 2 is a real class (with 2 classes it is the ignored sky id).
        share, support, counts = aligner(num_classes=3).project_semantics(
            seg, [meta], depth, return_counts=True)
        observed = support[0, 0] > 0
        self.assertGreater(observed.sum().item(), 0)
        torch.testing.assert_close(counts.sum(1, keepdim=True), support)
        torch.testing.assert_close(share[:, :, observed], (counts / support.clamp_min(1.0))[:, :, observed])
        self.assertEqual(counts.sum().item(), 3.0)   # three labeled pixels

    def test_two_camera_bilinear_counts_preserve_unequal_mass(self):
        seg = torch.zeros(1, 2, 4, 4, dtype=torch.long)
        depth = torch.zeros(1, 2, 4, 4)
        first = ((1, 1), (1, 2), (2, 1), (2, 2))
        second = ((1, 1), (1, 2))
        for row, column in first:
            seg[0, 0, row, column] = 1
            depth[0, 0, row, column] = 1
        for row, column in second:
            seg[0, 1, row, column] = 2
            depth[0, 1, row, column] = 1
        meta = dict(lidar2img=[np.eye(4), np.eye(4)], pad_shape=[(4, 4, 3)] * 2)
        share, support, counts = aligner(num_classes=3, splat_mode='bilinear').project_semantics(
            seg, [meta], depth, return_counts=True)
        # Each selected pixel contributes 0.25 to this cell: four pixels versus two.
        self.assertEqual(counts[0, 1, 5, 5].item(), 1.0)
        self.assertEqual(counts[0, 2, 5, 5].item(), 0.5)
        self.assertEqual(support[0, 0, 5, 5].item(), 1.5)
        self.assertEqual(share[0, 1, 5, 5].item(), 0.5)
        self.assertEqual(share[0, 2, 5, 5].item(), 0.25)
        torch.testing.assert_close(counts.sum(1, keepdim=True), support)

    def test_live_path_equals_cached_path(self):
        torch.manual_seed(0)
        model = aligner(semantic_type='share+logcount').eval()
        seg, depth, meta = two_pixel_maps()
        share, _, counts = model.project_semantics(seg, [meta], depth, return_counts=True)
        cached = semantic_input(share, counts, 'share+logcount', model.logcount_scale)
        with torch.no_grad():
            live = model(seg, [meta], depth_maps=depth)
            from_cache = model(None, [meta], semantic_prob=cached)
        for scale in live:
            torch.testing.assert_close(live[scale], from_cache[scale], rtol=0, atol=0)

    def test_log_share_applies_to_share_channels_only(self):
        model = aligner(semantic_type='share+logcount', use_log_share=True, log_share_eps=0.01)
        captured = []
        handle = model.prob_to_emb[0].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        x = torch.rand(1, 6, 8, 8)
        try:
            model.encode_prob(x)
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0][:, 3:], x[:, 3:])
        torch.testing.assert_close(captured[0][:, :3],
                                   torch.log1p(x[:, :3] / 0.01) / math.log1p(1 / 0.01))


class CacheLayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.projection = dict(bev_h=8, bev_w=8, num_classes=3)
        self.args = (self.projection, (900, 1600), (2.5, 170), {0: 0}, 'test-v1')
        self.meta = dict(filename=['/x/samples/CAM_FRONT/f.jpg'], lidar2img=[np.eye(4)],
                         metric3d_lidar2img_raw=[np.eye(4)], img_shape=[(480, 800, 3)],
                         pad_shape=[(480, 800, 3)], ori_shape=[(450, 800, 3)])

    def test_legacy_contract_has_no_layout_field(self):
        legacy = cache_contract(*self.args)
        self.assertNotIn('layout', legacy)
        self.assertEqual(legacy, cache_contract(*self.args, layout='share'))
        self.assertEqual(cache_contract(*self.args, layout='share+counts+support')['layout'],
                         'share+counts+support')
        with self.assertRaisesRegex(ValueError, 'layout'):
            cache_contract(*self.args, layout='counts')

    def test_counts_layout_round_trip_and_split(self):
        contract = cache_contract(*self.args, layout='share+counts+support')
        root = Path(self.tmp.name) / 'counts'
        cache = SemanticBEVCache(root, contract, create=True)
        self.assertEqual(cache.shape, (cache_channels(3, 'share+counts+support'), 8, 8))
        value = np.random.rand(9, 8, 8).astype(np.float32)
        cache.save('s', self.meta, value)
        cache.flush()
        loaded = SemanticBEVCache(root, contract, required_tokens=['s']).load('s', self.meta)
        np.testing.assert_array_equal(loaded, value)
        share, counts, support = split_cache_array(loaded, 4, 'share+counts+support')
        self.assertEqual((share.shape, counts.shape, support.shape), ((4, 8, 8), (4, 8, 8), (1, 8, 8)))
        with self.assertRaises(ValueError):
            cache.save('bad', self.meta, np.zeros((4, 8, 8), np.float32))

    def test_layout_mismatch_rejected(self):
        root = Path(self.tmp.name) / 'legacy'
        SemanticBEVCache(root, cache_contract(*self.args), create=True).flush()
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            SemanticBEVCache(root, cache_contract(*self.args, layout='share+counts+support'))


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_cache(self, layout, value):
        root = Path(self.tmp.name) / layout.replace('+', '_')
        root.mkdir()
        contract = {'projection': {}, **({} if layout == 'share' else {'layout': layout})}
        (root / 'manifest.json').write_text(json.dumps({'contract': contract, 'entries': {}}))
        np.save(root / 'tok.npy', value)
        return root

    def dataset(self, root, semantic_type):
        ds = SimpleNamespace(semantic_bev_cache_root=str(root), semantic_bev_cache_shape=(17, 4, 4))
        configure_semantic_type(ds, semantic_type, 1e5, True, str(root))
        return ds

    def test_manifest_check_mirrors_output_frame(self):
        legacy = self.make_cache('share', np.zeros((17, 4, 4), np.float32))
        with self.assertRaisesRegex(ValueError, 'share\\+counts\\+support'):
            self.dataset(legacy, 'share+logcount')
        with self.assertRaisesRegex(ValueError, 'semantic_type'):
            self.dataset(legacy, 'presence')
        self.assertEqual(self.dataset(legacy, 'share').semantic_bev_cache_layout, 'share')

    def test_load_from_counts_cache(self):
        value = np.random.rand(35, 4, 4).astype(np.float32)
        value[17:34] *= 1000
        root = self.make_cache('share+counts+support', value)
        load = CustomNuScenesDiffusionDataset_seg_depth_v2.load_semantic_bev
        logcount = load(self.dataset(root, 'share+logcount'), 'tok')
        share_only = load(self.dataset(root, 'share'), 'tok')
        self.assertEqual(logcount.shape, (34, 4, 4))
        np.testing.assert_array_equal(logcount[:17], value[:17])
        np.testing.assert_allclose(logcount[17:], np.log1p(value[17:34]) / math.log1p(1e5), rtol=1e-6)
        np.testing.assert_array_equal(share_only, value[:17])

    def test_legacy_load_unchanged(self):
        value = np.random.rand(17, 4, 4).astype(np.float32)
        root = self.make_cache('share', value)
        out = CustomNuScenesDiffusionDataset_seg_depth_v2.load_semantic_bev(self.dataset(root, 'share'), 'tok')
        np.testing.assert_array_equal(out, value)


if __name__ == '__main__':
    unittest.main()
