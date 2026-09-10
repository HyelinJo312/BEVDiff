"""Cache fidelity, invalidation and fail-closed behavior without raw map I/O."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mmcv import Config

import projects.mmdet3d_plugin
from projects.bevdiffuser.semantic_bev_cache import (
    SemanticBEVCache, cache_contract, validate_cache_config,
)


class SemanticBEVCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.projection = dict(bev_h=8, bev_w=8, num_classes=3)
        self.contract = cache_contract(self.projection, (900, 1600), (2.5, 170),
                                       {0: 0, 1: 1}, 'test-v1')
        self.meta = dict(filename=['/images/samples/CAM_FRONT/frame.jpg'],
            lidar2img=[np.eye(4)], metric3d_lidar2img_raw=[np.eye(4)],
            img_shape=[(480, 800, 3)], pad_shape=[(480, 800, 3)],
            ori_shape=[(450, 800, 3)])
        self.value = np.zeros((4, 8, 8), dtype=np.float32)
        self.value[1:3, 2, 2] = [.125, .25]
        self.cache = SemanticBEVCache(self.root, self.contract, create=True)
        self.cache.save('sample', self.meta, self.value)
        self.cache.flush()

    def test_float32_unknown_and_fractional_mass_round_trip(self):
        reader = SemanticBEVCache(self.root, self.contract, required_tokens=['sample'])
        value = reader.load('sample', self.meta)
        np.testing.assert_array_equal(value, self.value)
        np.testing.assert_array_equal(value.argmax(0), self.value.argmax(0))
        np.testing.assert_array_equal(value.sum(0) == 0, self.value.sum(0) == 0)
        self.assertEqual(value[:, 2, 2].sum(), .375)

    def test_contract_and_missing_coverage_rejected(self):
        changed = copy.deepcopy(self.contract)
        changed['projection']['pixel_stride'] = 2
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            SemanticBEVCache(self.root, changed)
        changed = copy.deepcopy(self.contract)
        changed['source_version'] = 'test-v2'
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            SemanticBEVCache(self.root, changed)
        changed = copy.deepcopy(self.contract)
        changed['projection_runtime']['allow_tf32'] = not changed['projection_runtime']['allow_tf32']
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            SemanticBEVCache(self.root, changed)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            SemanticBEVCache(self.root, self.contract, required_tokens=['missing'])
        with self.assertRaises(KeyError):
            self.cache.load('missing', self.meta)

    def test_geometry_change_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta['lidar2img'][0][0, 0] = .5
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            self.cache.load('sample', meta)

    def test_corruption_and_missing_file_rejected(self):
        altered = self.value.copy()
        altered[1, 2, 2] += .01
        np.save(self.root / 'sample.npy', altered)
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            self.cache.load('sample', self.meta)
        (self.root / 'sample.npy').unlink()
        with self.assertRaises(FileNotFoundError):
            self.cache.load('sample', self.meta)

    def test_invalid_values_and_unsafe_tokens_rejected(self):
        for value in (self.value.astype(np.float16), self.value[:, :2],
                      np.full_like(self.value, np.nan), np.full_like(self.value, -1)):
            with self.assertRaises(ValueError):
                self.cache.save('bad', self.meta, value)
        with self.assertRaises(ValueError):
            self.cache.save('../bad', self.meta, self.value)

    def test_config_override_cannot_use_stale_projection(self):
        settings = dict(use_semantic_bev_cache=True, depth_raw_shape=(900, 1600),
            depth_range=(2.5, 170), seg_id_remap={0: 0, 1: 1},
            semantic_bev_cache_source_version='test-v1',
            semantic_bev_cache_projection=self.projection)
        cfg = Config(dict(data=dict(train=settings),
            unet=dict(parameters=dict(seg_bev_aligner=self.projection))))
        validate_cache_config(cfg)
        cfg.unet.parameters.seg_bev_aligner.pixel_stride = 2
        with self.assertRaisesRegex(ValueError, 'differ'):
            validate_cache_config(cfg)
        cfg.data.train.use_semantic_bev_cache = False
        validate_cache_config(cfg)


if __name__ == '__main__':
    unittest.main()
