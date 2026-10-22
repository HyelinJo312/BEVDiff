"""Parity tests against the unchanged occupancy training implementation."""

import copy
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmdet3d.models.builder import build_head, build_model

import projects.mmdet3d_plugin  # noqa: F401
from projects.bevdiffuser.occ_task import (
    BEVFormerOccDiffusion, LoadOccGTForDiffusion,
)
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from projects.mmdet3d_plugin.datasets.pipelines.loading_occ import LoadOccGTFromFile

ROOT = Path(__file__).resolve().parents[1]


class OccDiffusionTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.baseline_cfg = Config.fromfile(str(ROOT / 'projects/configs/occ3d/bevformer_base_occ.py'))
        cls.diffusion_cfg = Config.fromfile(str(ROOT / 'projects/configs/bevdiffuser/bev_base_onlyseg_sam_v2_da3_occ.py'))
        cls.reference = build_head(copy.deepcopy(cls.baseline_cfg.model.pts_bbox_head))

    def head_with_reference_weights(self):
        head = build_head(copy.deepcopy(self.diffusion_cfg.model.occ_head))
        head.decoder.load_state_dict(self.reference.transformer.decoder.state_dict())
        head.predicter.load_state_dict(self.reference.transformer.predicter.state_dict())
        return head

    def test_logits_loss_gradients_and_batch_norm_match_reference(self):
        head = self.head_with_reference_weights()
        self.reference.train()
        head.train()
        # Non-square XY and B>1 expose layout bugs hidden by a 200x200 grid.
        bev = torch.randn(2, 256, 5, 7, requires_grad=True)
        reference_bev = bev.detach().clone().requires_grad_(True)
        tokens = reference_bev.flatten(2).transpose(1, 2)
        with patch.object(self.reference.transformer, 'get_bev_features', return_value=tokens):
            _, ref_logits = self.reference.transformer([reference_bev], None, None, 5, 7)
        outputs = head(bev)
        self.assertEqual(outputs['occ'].shape, (2, 7, 5, 16, 18))
        torch.testing.assert_close(outputs['occ'], ref_logits, rtol=0, atol=0)
        gt = torch.randint(0, 18, (2, 7, 5, 16), dtype=torch.uint8)
        camera_mask = torch.zeros_like(gt)
        actual = head.loss(gt, camera_mask, outputs)
        expected = self.reference.loss(gt, camera_mask, dict(occ=ref_logits))
        self.assertEqual(set(actual), {'loss_occ'})
        torch.testing.assert_close(actual['loss_occ'], expected['loss_occ'], rtol=0, atol=0)
        self.assertGreater(actual['loss_occ'].item(), 0)
        all_visible = head.loss(gt, torch.ones_like(gt), outputs)['loss_occ']
        torch.testing.assert_close(actual['loss_occ'], all_visible, rtol=0, atol=0)
        actual['loss_occ'].backward()
        expected['loss_occ'].backward()
        torch.testing.assert_close(bev.grad, reference_bev.grad, rtol=0, atol=0)
        for name, p in head.decoder.named_parameters():
            torch.testing.assert_close(p.grad, dict(self.reference.transformer.decoder.named_parameters())[name].grad,
                                       rtol=0, atol=0)
        for name, state in head.decoder.named_buffers():
            torch.testing.assert_close(state, dict(self.reference.transformer.decoder.named_buffers())[name], rtol=0, atol=0)
        torch.testing.assert_close(head.get_occ(outputs, []), self.reference.get_occ(dict(occ=ref_logits), []))

    def test_gt_loader_matches_baseline_and_keeps_free_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'scene' / 'sample' / 'labels.npz'
            path.parent.mkdir(parents=True)
            semantics = (np.arange(7 * 5 * 16) % 18).reshape(7, 5, 16).astype(np.uint8)
            np.savez(path, semantics=semantics, mask_camera=(semantics % 2).astype(bool),
                     mask_lidar=np.ones_like(semantics))
            reference = LoadOccGTFromFile(str(root))(dict(occ_gt_path='scene/sample/labels.npz'))
            actual = LoadOccGTForDiffusion(str(root))(dict(sample_idx='sample'))
            for key in ('voxel_semantics', 'mask_camera', 'mask_lidar'):
                np.testing.assert_array_equal(actual[key], reference[key])
                self.assertEqual(actual[key].dtype, reference[key].dtype)
            batch = collate([{key: actual[key] for key in ('voxel_semantics', 'mask_camera', 'mask_lidar')}] * 2,
                            samples_per_gpu=2)
            self.assertEqual(batch['voxel_semantics'].shape, (2, 7, 5, 16))
            self.assertEqual(batch['voxel_semantics'].max().item(), 17)
            with self.assertRaises(FileNotFoundError):
                LoadOccGTForDiffusion(str(root))(dict(sample_idx='missing'))

    def test_baseline_checkpoint_task_weights_and_native_roundtrip(self):
        model = build_model(copy.deepcopy(self.diffusion_cfg.model))
        reference_state = {
            'pts_bbox_head.transformer.' + key: value.clone()
            for key, value in self.reference.transformer.state_dict().items()
            if key.startswith(('decoder.', 'predicter.'))
        }
        # Test the training loader and stock PyTorch checkpoint loading.
        from mmcv.runner import load_state_dict
        logger = logging.Logger('occ_task_test')
        logger.addHandler(logging.NullHandler())
        load_state_dict(model, reference_state, strict=False, logger=logger)
        for key, value in model.occ_head.state_dict().items():
            torch.testing.assert_close(value, reference_state['pts_bbox_head.transformer.' + key], rtol=0, atol=0)
        incompatible = model.load_state_dict(reference_state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertFalse(any(key.startswith('occ_head.') for key in incompatible.missing_keys))
        # Native task state must win if a checkpoint contains both naming schemes.
        mixed_state = {key: torch.zeros_like(value) for key, value in reference_state.items()}
        mixed_state.update({'occ_head.' + key: value.clone() for key, value in model.occ_head.state_dict().items()})
        model.load_state_dict(mixed_state, strict=False)
        for key, value in model.occ_head.state_dict().items():
            torch.testing.assert_close(value, reference_state['pts_bbox_head.transformer.' + key], rtol=0, atol=0)
        recovered = build_head(copy.deepcopy(self.diffusion_cfg.model.occ_head))
        recovered.load_state_dict(model.occ_head.state_dict(), strict=True)
        # Given-BEV loss must not run an image backbone or encoder.
        model.requires_grad_(False)
        model.occ_head.requires_grad_(True)
        model.pts_bbox_head.bev_h, model.pts_bbox_head.bev_w = 5, 7
        bev = torch.randn(2, 256, 5, 7, requires_grad=True)
        tokens = bev.flatten(2).transpose(1, 2)
        for layout in (bev, tokens, tokens.transpose(0, 1)):
            torch.testing.assert_close(model._bev_embed_to_2d(layout), bev, rtol=0, atol=0)
        with patch.object(model, 'extract_feat', side_effect=AssertionError('Backbone was called')), \
             patch.object(model.pts_bbox_head, 'forward', side_effect=AssertionError('BEV encoder was called')):
            loss = model(return_loss=True, given_bev=tokens,
                         voxel_semantics=torch.zeros(2, 7, 5, 16, dtype=torch.long),
                         mask_camera=torch.zeros(2, 7, 5, 16))['loss_occ']
            loss.backward()
        self.assertIsNotNone(bev.grad)
        self.assertTrue(all(p.grad is None for p in model.pts_bbox_head.parameters()))
        # The pre-BEV methods are inherited without an override.
        for name in ('extract_feat', 'extract_img_feat', 'obtain_bev', 'obtain_history_bev', 'simple_test_bev'):
            self.assertIs(getattr(BEVFormerOccDiffusion, name), getattr(BEVFormer, name))


if __name__ == '__main__':
    unittest.main()
