"""Focused synthetic checks for v5 occupancy output-frame selection."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from mmcv import Config

import projects.mmdet3d_plugin
from projects.bevdiffuser.data_utils import Occ3DDiffusionNuScenesDataset
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner
from projects.bevdiffuser.semantic_bev_cache import SemanticBEVCache, cache_contract, geometry_digest
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import CustomCollect3D
from projects.mmdet3d_plugin.datasets.nuscenes_occ import NuSceneOcc


class OccupancyOutputFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def aligner(self, **overrides):
        settings = dict(bev_h=8, bev_w=8, pc_range=(-4, -4, -2, 4, 4, 2),
                        num_classes=2, emb_channels=8, channel_mult=(1, 1, 1),
                        depth_range=(.1, 10), splat_mode='nearest')
        settings.update(overrides)
        return SegBEVAligner(**settings)

    def meta(self, ego2lidar=None):
        return dict(filename=['samples/CAM_FRONT/test.jpg'], lidar2img=[np.eye(4)],
                    metric3d_lidar2img_raw=[np.eye(4)], pad_shape=[(4, 4, 3)],
                    ori_shape=[(4, 4, 3)], img_shape=[(4, 4, 3)],
                    ego2lidar=np.eye(4) if ego2lidar is None else ego2lidar)

    def maps(self, row=1, column=1, depth=1):
        seg = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        distance = torch.zeros(1, 1, 4, 4)
        seg[0, 0, row, column] = 1
        distance[0, 0, row, column] = depth
        return seg, distance

    def test_default_lidar_and_identity_ego(self):
        seg, depth = self.maps()
        default = self.aligner()
        explicit = self.aligner(output_frame='lidar')
        ego = self.aligner(output_frame='ego')
        original = default.project_semantics(seg, [self.meta()], depth)
        for model in (explicit, ego):
            actual = model.project_semantics(seg, [self.meta()], depth)
            for expected_array, actual_array in zip(original, actual):
                torch.testing.assert_close(expected_array, actual_array, rtol=0, atol=0)
        self.assertEqual(original[0].dtype, torch.float32)
        self.assertEqual(original[0].shape, (1, 3, 8, 8))
        self.assertEqual(original[0][0, 1, 5, 5].item(), 1.)
        self.assertEqual(set(default.state_dict()), set(ego.state_dict()))

    def test_rotation_translation_before_splat(self):
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        lidar2ego[:3, 3] = [1, -1, 0]
        seg, depth = self.maps()
        probability, _ = self.aligner(output_frame='ego').project_semantics(
            seg, [self.meta(np.linalg.inv(lidar2ego))], depth)
        self.assertEqual(probability[0, 1, 4, 4].item(), 1.)
        self.assertEqual(torch.count_nonzero(probability).item(), 1)

    def test_pitch_height_filter_uses_ego_xyz(self):
        angle = np.pi / 4
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = [[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        seg, depth = self.maps(row=0, column=1, depth=3)
        settings = dict(pc_range=(-10, -10, -2, 10, 10, 2))
        lidar, _ = self.aligner(**settings).project_semantics(seg, [self.meta()], depth)
        ego, _ = self.aligner(output_frame='ego', **settings).project_semantics(
            seg, [self.meta(np.linalg.inv(lidar2ego))], depth)
        self.assertEqual(lidar.sum().item(), 0)
        self.assertGreater(ego.sum().item(), 0)

    def test_invalid_ego_calibration(self):
        seg, depth = self.maps()
        model = self.aligner(output_frame='ego')
        for calibration in (np.full((4, 4), np.nan), np.ones((3, 3))):
            with self.assertRaises(ValueError):
                model.project_semantics(seg, [self.meta(calibration)], depth)
        with self.assertRaises(RuntimeError):
            model.project_semantics(seg, [self.meta(np.zeros((4, 4)))], depth)
        share, _ = model.project_semantics(seg, [self.meta(np.diag([2., 1., 1., 1.]))], depth)
        self.assertTrue(torch.isfinite(share).all())
        meta = self.meta()
        del meta['ego2lidar']
        with self.assertRaisesRegex(ValueError, 'ego2lidar'):
            model.project_semantics(seg, [meta], depth)
        with self.assertRaisesRegex(ValueError, 'output_frame'):
            self.aligner(output_frame='world')

    def test_geometry_digest_and_cache_frame(self):
        meta = self.meta()
        changed = copy.deepcopy(meta)
        changed['ego2lidar'][0, 3] = 1
        self.assertEqual(geometry_digest(meta), geometry_digest(changed))
        self.assertNotEqual(geometry_digest(meta, 'ego'), geometry_digest(changed, 'ego'))
        with tempfile.TemporaryDirectory() as directory:
            projection = dict(bev_h=8, bev_w=8, num_classes=2, output_frame='ego')
            contract = cache_contract(projection, (4, 4), (.1, 10), {}, 'ego-test')
            self.assertEqual(contract['projection']['output_frame'], 'ego')
            cache = SemanticBEVCache(directory, contract, create=True)
            cache.save('sample', meta, np.zeros((3, 8, 8), dtype=np.float32))
            cache.flush()
            with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
                cache.load('sample', changed)
            lidar_contract = copy.deepcopy(contract)
            lidar_contract['projection']['output_frame'] = 'lidar'
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                SemanticBEVCache(directory, lidar_contract)

    def test_occupancy_config_and_launcher(self):
        root = Path(__file__).resolve().parents[1]
        config = Config.fromfile(str(root / 'projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py'))
        self.assertEqual(config.unet.parameters.seg_bev_aligner.output_frame, 'ego')
        self.assertEqual(config.semantic_bev_cache_source_version, 'semantic_bev_occ_ego')
        self.assertTrue(config.semantic_bev_cache_root.endswith('semantic_bev_occ_v2'))
        launcher = (root / 'projects/bevdiffuser/build_semantic_bev_cache_occ.sh').read_text()
        self.assertIn('bev_onlyseg_sam_v2_metric3d_occ.py', launcher)
        self.assertNotIn('bev_tiny_onlyseg_sam_v2_metric3d_occ.py', launcher)
        self.assertIn('semantic_bev_occ_v2', launcher)
        self.assertIn('semantic_bev_occ_ego', launcher)
        self.assertIn('--devices "${devices[@]}"', launcher)

    def test_occupancy_collector_retains_raw_projection(self):
        collector = CustomCollect3D(keys=['img'])
        self.assertNotIn('metric3d_lidar2img_raw', collector.meta_keys)
        self.assertEqual(Occ3DDiffusionNuScenesDataset._retain_raw_projection(collector), 1)
        self.assertIn('metric3d_lidar2img_raw', collector.meta_keys)

    def test_occupancy_cache_constructor_checks_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            arguments = dict(use_semantics=True, use_depth=True,
                             use_semantic_bev_cache=True, semantic_bev_cache_root=directory)
            with mock.patch.object(NuSceneOcc, '__init__', return_value=None):
                manifest.write_text(json.dumps(dict(contract=dict(projection=dict(output_frame='ego')))))
                Occ3DDiffusionNuScenesDataset(**arguments, semantic_bev_cache_output_frame='ego')
                with self.assertRaisesRegex(ValueError, 'output frame'):
                    Occ3DDiffusionNuScenesDataset(**arguments, semantic_bev_cache_output_frame='lidar')
                manifest.write_text(json.dumps(dict(contract=dict(projection={}))))
                Occ3DDiffusionNuScenesDataset(**arguments, semantic_bev_cache_output_frame='lidar')
                with self.assertRaisesRegex(ValueError, 'output frame'):
                    Occ3DDiffusionNuScenesDataset(**arguments, semantic_bev_cache_output_frame='ego')


if __name__ == '__main__':
    unittest.main()
