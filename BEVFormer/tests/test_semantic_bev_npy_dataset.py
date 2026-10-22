"""Direct-cache integration with temporal metadata and MMCV batching."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC, collate

import projects.mmdet3d_plugin  # Register plugin before importing dataset classes.
from projects.bevdiffuser.data_utils import (
    CustomNuScenesDataset,
    CustomNuScenesDiffusionDataset_seg_depth_v2 as Dataset,
    NuSceneOcc,
    Occ3DDiffusionNuScenesDataset as OccDataset,
)


class SemanticBEVNpyDatasetTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.value = np.zeros((17, 50, 50), dtype=np.float32)
        self.value[2:4, 10, 20] = (.125, .25)
        np.save(self.root / 'current.npy', self.value)

    def dataset(self, **overrides):
        options = dict(pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                       use_semantics=True, use_depth=True,
                       semantic_path=None, depth_path=None,
                       use_semantic_bev_cache=True,
                       semantic_bev_cache_root=self.root,
                       semantic_bev_cache_shape=self.value.shape)
        options.update(overrides)
        # Isolate external annotations/RGB loading; exercise the actual v2 init.
        with patch.object(CustomNuScenesDataset, '__init__', return_value=None):
            return Dataset(**options)

    def item(self, dataset, metadata):
        with patch.object(CustomNuScenesDataset, '__getitem__',
                          return_value={'img_metas': metadata}), \
                patch.object(dataset, 'load_segmaps', side_effect=AssertionError('SAM access')), \
                patch.object(dataset, 'load_depth_from_filenames', side_effect=AssertionError('depth access')):
            return dataset[123]  # The returned token, not the requested index, selects the cache.

    def test_temporal_current_token_and_collation_without_manifest_or_raw_maps(self):
        dataset = self.dataset()
        meta = DC({0: {'filename': [], 'sample_idx': 'missing_history'},
                   2: {'filename': [], 'sample_idx': 'current'}}, cpu_only=True)
        item = self.item(dataset, meta)
        self.assertNotIn('seg_maps', item)
        self.assertNotIn('depth_maps', item)
        self.assertFalse((self.root / 'manifest.json').exists())
        torch.testing.assert_close(item['semantic_bev_probabilities'].data,
                                   torch.from_numpy(self.value), rtol=0, atol=0)
        batch = collate([item, item], samples_per_gpu=2)
        probabilities = torch.stack(batch['semantic_bev_probabilities'].data[0], dim=0)
        self.assertEqual(tuple(probabilities.shape), (2, 17, 50, 50))
        self.assertEqual(probabilities[0, :, 10, 20].sum().item(), .375)
        self.assertEqual(probabilities[0, :, 0, 0].sum().item(), 0)

    def test_test_augmentation_and_stale_manifest_are_supported(self):
        (self.root / 'manifest.json').write_text('not even valid JSON')
        dataset = self.dataset()
        item = self.item(dataset, [DC({'filename': [], 'sample_idx': 'current'}, cpu_only=True)])
        self.assertEqual(item['semantic_bev_probabilities'].data.dtype, torch.float32)

    def test_occ_cache_uses_direct_numpy_without_contract_metadata(self):
        (self.root / 'manifest.json').write_text('not even valid JSON')

        def initialize_base(instance, *args, **kwargs):
            instance.pipeline = object()

        options = dict(
            use_semantics=True,
            use_depth=True,
            semantic_path=None,
            depth_path=None,
            use_semantic_bev_cache=True,
            semantic_bev_cache_root=self.root,
            semantic_bev_cache_shape=self.value.shape,
        )
        with patch.object(NuSceneOcc, '__init__', initialize_base), \
                patch.object(OccDataset, '_retain_raw_projection', return_value=True):
            dataset = OccDataset(**options)

        metadata = DC({'filename': [], 'sample_idx': 'current'}, cpu_only=True)
        with patch.object(dataset, 'get_unconditioned_item',
                          return_value={'img_metas': metadata}):
            item = dataset[0]
        torch.testing.assert_close(
            item['semantic_bev_probabilities'].data,
            torch.from_numpy(self.value),
            rtol=0,
            atol=0,
        )

    def test_missing_file_does_not_fall_back_to_raw_maps(self):
        dataset = self.dataset()
        with self.assertRaisesRegex(FileNotFoundError, 'missing.npy'):
            self.item(dataset, DC({'filename': [], 'sample_idx': 'missing'}))

    def test_wrong_shape_and_label_arrays_report_file(self):
        dataset = self.dataset()
        for invalid in (self.value[:, :25], self.value[0], self.value.astype(np.int64)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                np.save(self.root / 'invalid.npy', invalid)
                with self.assertRaisesRegex(ValueError, 'invalid.npy'):
                    dataset.load_semantic_bev('invalid')

    def test_floating_conversion_preserves_values(self):
        np.save(self.root / 'half.npy', self.value.astype(np.float16))
        actual = self.dataset().load_semantic_bev('half')
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, self.value)

    def test_previous_projection_config_needs_no_source_version(self):
        dataset = self.dataset(semantic_bev_cache_shape=None,
                               semantic_bev_cache_projection=dict(num_classes=16, bev_h=50, bev_w=50))
        np.testing.assert_array_equal(dataset.load_semantic_bev('current'), self.value)

    def test_temporal_metadata_order_and_latest_valid(self):
        dataset = object.__new__(Dataset)
        dataset._info_by_token = {
            token: dict(lidar2ego_rotation=[1, 0, 0, 0], lidar2ego_translation=[0, 0, 0],
                        ego2global_rotation=[1, 0, 0, 0], ego2global_translation=[x, 0, 0],
                        timestamp=int(x * 1e6))
            for token, x in (('a', 1), ('b', 2))
        }
        metas = DC({2: DC({'filename': [], 'sample_idx': 'a'}),
                    0: DC({'filename': [], 'sample_idx': 'a'}),
                    1: DC({'filename': [], 'sample_idx': 'b'})}, cpu_only=True)
        frames = dataset._temporal_metas(metas)
        self.assertEqual([frame['sample_idx'] for frame in frames], ['a', 'b', 'a'])
        with patch.object(dataset, 'load_semantic_bev',
                          side_effect=lambda token: np.full((2, 2, 2), 1 if token == 'a' else 2,
                                                             dtype=np.float32)):
            probabilities, transforms, timestamps, valid = dataset._temporal_semantic_bev(frames)
        np.testing.assert_array_equal(probabilities[:, 0, 0, 0], [1, 2, 1])
        np.testing.assert_array_equal(transforms[:, 0, 3], [1, 2, 1])
        np.testing.assert_array_equal(timestamps, [1, 2, 1])
        np.testing.assert_array_equal(valid, [False, True, True])

    def test_raw_remap_depth_sanitation_and_affine_warp(self):
        dataset = object.__new__(Dataset)
        dataset.depth_raw_shape = (2, 3)
        dataset.depth_range = (2.5, 10)
        dataset.seg_id_remap = {16: 2, 17: 2}
        dataset.depth_path = str(self.root)
        camera = self.root / 'samples' / 'CAM_FRONT'
        camera.mkdir(parents=True)
        np.array([[16, 1, -1], [0, 17, 2]], dtype=np.int8).tofile(camera / 'a_mask.bin')
        np.save(camera / 'a.npy', np.array([[3, np.nan, 5], [0, 4, 100]], dtype=np.float32))
        projection = np.eye(4)
        augmented = projection.copy()
        augmented[0, 2] = 1  # one-pixel image-plane translation
        meta = dict(metric3d_lidar2img_raw=[projection], lidar2img=[augmented],
                    ori_shape=[(2, 4, 3)], pad_shape=[(3, 4, 3)])
        filename = ['samples/CAM_FRONT/a.jpg']
        seg = dataset.load_segmaps(filename, meta, str(self.root))
        depth = dataset.load_depth_from_filenames(filename, meta)
        np.testing.assert_array_equal(seg.numpy()[0],
                                      [[-1, 2, 1, -1], [-1, 0, 2, 2], [-1, -1, -1, -1]])
        np.testing.assert_array_equal(depth.numpy()[0],
                                      [[0, 3, 0, 5], [0, 0, 4, 0], [0, 0, 0, 0]])

    def test_live_mode_uses_raw_loaders(self):
        dataset = self.dataset()
        dataset.use_semantic_bev_cache = False
        meta = DC({'filename': [], 'sample_idx': 'current'})
        with patch.object(CustomNuScenesDataset, '__getitem__', return_value={'img_metas': meta}), \
                patch.object(dataset, 'load_segmaps', return_value=torch.ones(1)) as seg, \
                patch.object(dataset, 'load_depth_from_filenames', return_value=torch.ones(1)) as depth, \
                patch.object(dataset, 'load_semantic_bev', side_effect=AssertionError('cache access')):
            item = dataset[0]
        seg.assert_called_once()
        depth.assert_called_once()
        self.assertIn('seg_maps', item)
        self.assertIn('depth_maps', item)
        self.assertNotIn('semantic_bev_probabilities', item)


if __name__ == '__main__':
    unittest.main()
