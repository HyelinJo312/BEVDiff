"""CPU-only checks for layout-occupancy teacher frame and range alignment."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from projects.bevdiffuser.bev_frame_alignment import TeacherBEVAlignment, masked_bev_mse
from test_layout_occ_gradient_monitor import make_model
from projects.bevdiffuser.data_utils import (
    CustomNuScenesDiffusionDataset_layout, Occ3DLayoutDiffusionNuScenesDataset)


def alignment(source_range=(-2, -2, -1, 2, 2, 1),
              teacher_range=(-2, -2, -1, 2, 2, 1), teacher_size=(4, 4), **kwargs):
    return TeacherBEVAlignment(source_range, teacher_range, teacher_size, **kwargs)


def meta(matrix=None):
    return {'ego2lidar': torch.eye(4).numpy() if matrix is None else matrix}


class FrameAlignmentTests(unittest.TestCase):
    def test_identity_batch_nonsquare_and_range_resampling(self):
        aligner = alignment(source_range=(0, 0, -1, 6, 4, 1),
                            teacher_range=(0, 0, -1, 6, 4, 1),
                            teacher_size=(4, 6))
        source = torch.arange(48, dtype=torch.float32).reshape(2, 1, 4, 6)
        teacher, support, transform = aligner.to_teacher(source, [meta(), meta()])
        output, valid = aligner.to_student(teacher, support, [meta(), meta()], (4, 6), transform)
        torch.testing.assert_close(teacher, source, rtol=0, atol=1e-6)
        torch.testing.assert_close(output, source, rtol=0, atol=1e-6)
        self.assertTrue(valid.all())

        resampler = alignment(source_range=(0, 0, -1, 6, 4, 1),
                              teacher_range=(0, 0, -1, 6, 4, 1),
                              teacher_size=(8, 12))
        upsampled, _, _ = resampler.to_teacher(source, [meta(), meta()])
        self.assertEqual(tuple(upsampled.shape), (2, 1, 8, 12))
        self.assertEqual(upsampled.dtype, source.dtype)
        wider = alignment(source_range=(0, 0, -1, 6, 4, 1),
                          teacher_range=(0, 0, -1, 12, 4, 1),
                          teacher_size=(4, 12))
        expanded, expanded_support, _ = wider.to_teacher(source, [meta(), meta()])
        torch.testing.assert_close(expanded[..., :6], source, rtol=0, atol=1e-6)
        self.assertTrue(expanded_support[..., :6].all())
        self.assertFalse(expanded_support[..., 6:].any())

    def test_integer_translation_and_quarter_turn_asymmetric_marker(self):
        source = torch.zeros(1, 1, 4, 4)
        source[0, 0, 1, 2] = 7
        translated = torch.eye(4)
        translated[0, 3] = 1
        shifted, _, _ = alignment().to_teacher(source, [meta(translated)])
        self.assertEqual(shifted[0, 0, 1, 3].item(), 7)
        turned = torch.eye(4)
        turned[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
        rotated, _, _ = alignment().to_teacher(source, [meta(turned)])
        self.assertEqual(rotated[0, 0, 2, 2].item(), 7)

    def test_lidar_bbox_and_warped_feature_share_teacher_cell(self):
        bounds = (-3, -2, -1, 3, 2, 1)
        source = torch.zeros(1, 1, 4, 6)
        source[0, 0, 1, 3] = 1
        transform = torch.eye(4)
        transform[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
        transform[0, 3] = 1
        teacher, _, _ = alignment(bounds, bounds, (4, 6)).to_teacher(
            source, [meta(transform)])
        dataset = Occ3DLayoutDiffusionNuScenesDataset.__new__(
            Occ3DLayoutDiffusionNuScenesDataset)
        dataset.layout_pc_range = bounds
        lidar_box = torch.tensor([[1.5, 0.5, 0., 1., 1., 1., 0., 0., 0.]])
        normalized = dataset.normalize_layout_bbox(lidar_box)
        box_x = int(torch.floor((normalized[0, 0] + 0.5) * 6))
        box_y = int(torch.floor((normalized[0, 1] + 0.5) * 4))
        marker_y, marker_x = divmod(teacher[0, 0].argmax().item(), 6)
        self.assertEqual((marker_y, marker_x), (2, 4))
        self.assertEqual((marker_y, marker_x), (box_y, box_x))

    def test_planar_inverse_uses_same_affine_with_roll_pitch(self):
        matrix = torch.eye(4)
        pitch = torch.tensor(0.08)
        roll = torch.tensor(-0.04)
        rotate_y = torch.tensor([[torch.cos(pitch), 0., torch.sin(pitch)],
                                 [0., 1., 0.],
                                 [-torch.sin(pitch), 0., torch.cos(pitch)]])
        rotate_x = torch.tensor([[1., 0., 0.],
                                 [0., torch.cos(roll), -torch.sin(roll)],
                                 [0., torch.sin(roll), torch.cos(roll)]])
        matrix[:3, :3] = rotate_y @ rotate_x
        matrix[:2, 3] = torch.tensor([0.2, -0.3])
        aligner = alignment(ego_plane_z=0.5)
        planar, offset, bound = aligner._planar_transform([meta(matrix)], 1, torch.device('cpu'))
        ego = torch.tensor([1.2, -0.7], dtype=torch.float64)
        lidar = planar[0] @ ego + offset[0]
        recovered = torch.linalg.solve(planar[0], lidar - offset[0])
        torch.testing.assert_close(recovered, ego, rtol=0, atol=1e-14)
        expected_bound = torch.linalg.vector_norm(matrix[:2, 2].double()) * 1.5
        torch.testing.assert_close(bound[0], expected_bound)

    def test_missing_bad_calibration_and_overlap_fail(self):
        source = torch.zeros(1, 1, 4, 4)
        with self.assertRaises(KeyError):
            alignment().to_teacher(source, [{}])
        for matrix in (torch.zeros(4, 4), torch.full((4, 4), float('nan'))):
            with self.assertRaises(ValueError):
                alignment().to_teacher(source, [meta(matrix)])
        singular = torch.eye(4)
        singular[1, :2] = singular[0, :2]
        with self.assertRaises(ValueError):
            alignment().to_teacher(source, [meta(singular)])
        scaled = torch.eye(4)
        scaled[0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'rigid'):
            alignment().to_teacher(source, [meta(scaled)])
        far = torch.eye(4)
        far[0, 3] = 100
        teacher, support, transform = alignment().to_teacher(source, [meta(far)])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            alignment().to_student(teacher, support, [meta(far)], (4, 4), transform)

    def test_strict_boundary_support_and_masked_kd_normalization(self):
        aligner = alignment(source_range=(0, 0, -1, 4, 4, 1),
                            teacher_range=(-0.5, 0, -1, 3.5, 4, 1))
        source = torch.ones(1, 2, 4, 4)
        teacher, support, transform = aligner.to_teacher(source, [meta()])
        self.assertFalse(support[0, 0, 0, 0])
        self.assertTrue(support[0, 0, 0, 1])
        output, valid = aligner.to_student(teacher, support, [meta()], (4, 4), transform)
        self.assertFalse(valid[0, 0, 0, 0])
        self.assertTrue(valid.any())
        student = torch.zeros_like(output, requires_grad=True)
        loss = masked_bev_mse(student, torch.ones_like(output) * 2, valid)
        torch.testing.assert_close(loss, torch.tensor(4.0))
        padded_teacher = torch.ones_like(output) * 2
        padded_teacher[~valid.expand_as(padded_teacher)] = float('nan')
        torch.testing.assert_close(masked_bev_mse(student, padded_teacher, valid), loss)
        loss.backward()
        self.assertEqual(student.grad[~valid.expand_as(student)].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            masked_bev_mse(student, output[:, :1], valid)

    def test_output_only_support_masks_padded_occupancy_cells(self):
        aligner = alignment(source_range=(0, 0, -1, 4, 4, 1),
                            teacher_range=(1, 0, -1, 5, 4, 1),
                            teacher_size=(8, 8))
        native = torch.ones(1, 2, 8, 8)
        output, valid = aligner.to_student(
            native, torch.ones(1, 1, 8, 8, dtype=torch.bool), [meta()], (4, 4))
        self.assertEqual(tuple(output.shape), (1, 2, 4, 4))
        self.assertFalse(valid[..., 0].any())
        self.assertTrue(valid[..., 1:].all())
        student = torch.zeros_like(output, requires_grad=True)
        loss = masked_bev_mse(student, output, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(student.grad[..., 0].abs().sum().item(), 0)
        self.assertGreater(student.grad[..., 1:].abs().sum().item(), 0)

    def test_layout_bbox_matches_original_formula_and_sentinels(self):
        teacher_range = (-51.2, -51.2, -5, 51.2, 51.2, 3)
        dataset = Occ3DLayoutDiffusionNuScenesDataset.__new__(
            Occ3DLayoutDiffusionNuScenesDataset)
        dataset.pc_range = (-40, -40, -1, 40, 40, 5.4)
        dataset.layout_pc_range = teacher_range
        dataset.num_bboxes = 3
        dataset.num_layout_classes = 12
        boxes = torch.tensor([[10., -7., 1., 4., 2., 1.5, 0.4, 2., -1.]])
        reference = CustomNuScenesDiffusionDataset_layout.__new__(
            CustomNuScenesDiffusionDataset_layout)
        reference.pc_range = teacher_range
        expected = reference.normalize_bbox(boxes)
        expected[..., :2] -= 0.5
        actual = dataset.normalize_layout_bbox(boxes)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[..., 6:], boxes[..., 6:])
        data = dict(gt_labels_3d=DC(torch.tensor([2])),
                    gt_bboxes_3d=DC(LiDARInstance3DBoxes(boxes, box_dim=9)))
        layout = dataset.get_layout_info(data)
        torch.testing.assert_close(layout['layout_obj_bboxes'].data[1], expected[0])
        torch.testing.assert_close(
            layout['layout_obj_bboxes'].data[0],
            torch.tensor([0., 0., 0., 1., 1., 1., 0., 0., 0.]))
        self.assertEqual(layout['layout_obj_classes'].data.tolist(), [10, 2, 11])
        self.assertEqual(layout['layout_obj_is_valid'].data.tolist(), [1., 1., 0.])

    def test_config_propagation_and_preflight(self):
        from tools.train_layout_occ import validate_teacher_alignment_config
        path = Path(__file__).resolve().parents[1] / 'projects/configs/diff_bevformer/layout_base_occ.py'
        cfg = Config.fromfile(str(path), import_custom_modules=False)
        validate_teacher_alignment_config(cfg)
        for split in ('train', 'val', 'test'):
            self.assertEqual(cfg.data[split].layout_pc_range, cfg.teacher_point_cloud_range)
        self.assertEqual(cfg.layout_dataset_args.layout_pc_range, cfg.teacher_point_cloud_range)
        self.assertEqual(cfg.train_pipeline[4].point_cloud_range, cfg.teacher_point_cloud_range)
        bad = copy.deepcopy(cfg)
        bad.data.val.layout_pc_range = bad.point_cloud_range
        with self.assertRaisesRegex(ValueError, 'val layout'):
            validate_teacher_alignment_config(bad)
        bad = copy.deepcopy(cfg)
        bad.bev_diffuser_cfg.unet_cfg.parameters.image_size = 50
        with self.assertRaisesRegex(ValueError, 'UNet image_size'):
            validate_teacher_alignment_config(bad)
        bad = copy.deepcopy(cfg)
        bad.model_target.pts_bbox_head.bbox_coder.pc_range = bad.point_cloud_range
        with self.assertRaisesRegex(ValueError, 'bbox coder range'):
            validate_teacher_alignment_config(bad)
        bad = copy.deepcopy(cfg)
        bad.model_target.pts_bbox_head.transformer.encoder.type = 'BEVFormerEncoderOcc'
        with self.assertRaisesRegex(ValueError, 'native detection transformer'):
            validate_teacher_alignment_config(bad)
        bad = copy.deepcopy(cfg)
        bad.model_target.pts_bbox_head.transformer.embed_dims = 128
        with self.assertRaisesRegex(ValueError, 'channels'):
            validate_teacher_alignment_config(bad)
        bad = copy.deepcopy(cfg)
        bad.data.train.pipeline[4].point_cloud_range = bad.point_cloud_range
        with self.assertRaisesRegex(ValueError, 'object filter'):
            validate_teacher_alignment_config(bad)

    def test_tiny_detector_teacher_input_frozen_gradients_and_monitor_invariance(self):
        class RecordingTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.seen = None
                self.grad_enabled = None

            def forward(self, teacher_input, condition, grad_fn=None):
                self.seen = teacher_input.detach().clone()
                self.grad_enabled = torch.is_grad_enabled()
                return teacher_input * self.weight

        config = dict(source_pc_range=(0, 0, -1, 3, 2, 1),
                      teacher_pc_range=(1, 0, -1, 4, 2, 1),
                      teacher_bev_size=(4, 6))
        model = make_model()
        model.teacher_bev_alignment = TeacherBEVAlignment(**config)
        monitored = copy.deepcopy(model)
        monitored._gradient_monitor_context = object()
        baseline_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        monitored_optimizer = torch.optim.SGD(monitored.parameters(), lr=0.01)
        teacher = RecordingTeacher().requires_grad_(False)
        features = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 24
        target = torch.arange(96, dtype=torch.float32).reshape(1, 4, 4, 6).requires_grad_()
        conditions = dict(layout_obj_classes=torch.zeros(1, 1),
                          layout_obj_bboxes=torch.zeros(1, 1, 9),
                          layout_obj_is_valid=torch.ones(1, 1))

        def step(current):
            return current.forward_pts_train(
                features, None, None, torch.zeros(1, 2, 2, 3), None,
                [meta()], bev_diffuser=teacher, bev_target=target,
                **conditions)

        rng = torch.random.get_rng_state()
        with patch.object(model.teacher_bev_alignment, 'to_teacher',
                          side_effect=AssertionError('input transport used')):
            baseline = step(model)
        baseline_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(rng)
        with patch.object(monitored.teacher_bev_alignment, 'to_teacher',
                          side_effect=AssertionError('input transport used')):
            probed = step(monitored)
        torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng)
        torch.testing.assert_close(baseline['loss_bev'], probed['loss_bev'])
        self.assertEqual(tuple(teacher.seen.shape), (1, 4, 4, 6))
        torch.testing.assert_close(teacher.seen, target.detach())
        self.assertFalse(teacher.grad_enabled)
        self.assertFalse(teacher.seen.requires_grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(teacher.weight.grad)
        self.assertTrue(torch.isfinite(baseline['loss_bev']))
        self.assertIsNotNone(monitored._gradient_monitor_pending)
        baseline_loss = sum(baseline.values())
        probed_loss = sum(probed.values())
        baseline_loss.backward()
        probed_loss.backward()
        for left, right in zip(model.parameters(), monitored.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
        baseline_optimizer.step()
        monitored_optimizer.step()
        for left, right in zip(model.parameters(), monitored.parameters()):
            torch.testing.assert_close(left, right)
        self.assertIsNone(teacher.weight.grad)


if __name__ == '__main__':
    unittest.main()
