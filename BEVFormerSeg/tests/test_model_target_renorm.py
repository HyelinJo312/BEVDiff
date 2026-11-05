"""Check the detection model_target wiring for the det->seg run without CUDA plugins."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'tools/train_onlyseg.py'
DETECTOR = ROOT / 'projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer.py'
CONFIG = ROOT / 'projects/configs/diff_bevformer/bev_tiny_det_semantic_onlyseg_sam_metric3d_det.py'
# Stage-1 config of the teacher results/bevdiffuser/det/BEVDiffuser_tiny_onlyseg_sam3_metric3d.
TEACHER_CONFIG = ROOT.parent / 'BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'
TEACHER_RUN_DIR = ROOT.parent / 'BEVFormer/projects/bevdiffuser'  # Stage-1 cwd for relative paths

CAFFE = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
TORCHVISION = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


def load_definition(source, name, scope):
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope[name]


ImageRenormTarget = load_definition(
    DETECTOR, 'ImageRenormTarget', {'torch': torch, 'np': np, 'F': F})


def scales(pipeline):
    found = []
    for step in pipeline:
        if step['type'] == 'RandomScaleImageMultiViewImage':
            found.append(list(step['scales']))
        found += scales(step.get('transforms', []))
    return found


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile(str(CONFIG))

    def test_target_is_detection_bevformer_with_student_encoder(self):
        cfg = self.cfg
        self.assertEqual(cfg.model_target.type, 'BEVFormer')
        self.assertEqual(cfg.model_target.pts_bbox_head.task, {'seg': False, 'det': True})
        self.assertEqual(cfg.model.pts_bbox_head.task, {'seg': True, 'det': False})
        for key in ('img_backbone', 'img_neck'):
            self.assertEqual(cfg.model_target[key], cfg.model[key])
        for key in ('bev_h', 'bev_w', 'positional_encoding'):
            self.assertEqual(cfg.model_target.pts_bbox_head[key], cfg.model.pts_bbox_head[key])
        self.assertEqual(cfg.model_target.pts_bbox_head.transformer.encoder,
                         cfg.model.pts_bbox_head.transformer.encoder)
        self.assertTrue(cfg.load_from.endswith('BEVFormer/ckpts/bevformer_tiny_epoch_24.pth'))

    def test_teacher_condition_matches_its_stage1_config(self):
        cfg, teacher = self.cfg, Config.fromfile(str(TEACHER_CONFIG))
        self.assertEqual(cfg.unet.type.split('.')[-2:], teacher.unet.type.split('.')[-2:])
        self.assertEqual(cfg.unet.parameters, teacher.unet.parameters)
        self.assertEqual(cfg.bev_diffuser_cfg.unet_cfg, cfg.unet)
        self.assertEqual(cfg.seg_id_remap, teacher.seg_id_remap)
        self.assertEqual(tuple(cfg.data.train.semantic_bev_cache_shape),
                         tuple(teacher.data.train.semantic_bev_cache_shape))
        self.assertEqual(cfg.data.train.semantic_type, 'share')
        self.assertEqual((ROOT / cfg.semantic_bev_cache_root).resolve(),
                         (TEACHER_RUN_DIR / teacher.semantic_bev_cache_root).resolve())

    def test_target_gets_torchvision_images_at_student_scale(self):
        cfg = self.cfg
        self.assertEqual(dict(cfg.img_norm_cfg), CAFFE)
        self.assertEqual(dict(cfg.model_target_img_norm_cfg), TORCHVISION)
        self.assertEqual(cfg.model_target_img_rescale, 1.0)
        self.assertEqual(scales(cfg.train_pipeline), [[0.8]])
        self.assertEqual(scales(cfg.test_pipeline), [[0.8]])


class RenormTests(unittest.TestCase):
    def test_student_input_is_mapped_to_checkpoint_input(self):
        cfg = Config.fromfile(str(CONFIG))
        raw_bgr = torch.rand(1, 2, 6, 3, 4, 5) * 255  # (B, queue, cams, C, H, W) as loaded by mmcv
        bgr_mean = torch.tensor(CAFFE['mean']).view(3, 1, 1)
        student = raw_bgr - bgr_mean
        raw_rgb = raw_bgr.flip(-3)
        expected = (raw_rgb - torch.tensor(TORCHVISION['mean']).view(3, 1, 1)) \
            / torch.tensor(TORCHVISION['std']).view(3, 1, 1)

        head = object()
        received = {}

        class Recorder(torch.nn.Module):
            pts_bbox_head = head

            def forward(self, *args, **kwargs):
                received.update(kwargs)
                return 'bev'

        target = ImageRenormTarget(Recorder(), cfg.img_norm_cfg, cfg.model_target_img_norm_cfg,
                                   rescale=cfg.model_target_img_rescale)
        metas = [{0: {'ori_shape': [(4, 5, 3)]}}]
        self.assertEqual(target(return_loss=False, only_bev=True, img=student, img_metas=metas), 'bev')
        torch.testing.assert_close(received['img'], expected)
        self.assertIs(received['img_metas'], metas)
        self.assertEqual((received['return_loss'], received['only_bev']), (False, True))
        self.assertIs(target.pts_bbox_head, head)


class BuildTargetTests(unittest.TestCase):
    def build(self, cfg, model=None):
        target = Mock()
        scope = {
            'copy': copy,
            'build_model': Mock(return_value=target),
            'load_checkpoint': Mock(),
            'ImageRenormTarget': Mock(return_value=Mock()),
        }
        build = load_definition(TRAIN, 'build_model_target', scope)
        return build(cfg, model or torch.nn.Linear(2, 2)), scope, target

    def test_explicit_target_loads_strictly_and_is_renormalized(self):
        cfg = Config(dict(model_target=dict(type='BEVFormer'), load_from='det.pth',
                          img_norm_cfg=CAFFE, model_target_img_norm_cfg=TORCHVISION,
                          model_target_img_rescale=1.0))
        result, scope, target = self.build(cfg)
        scope['build_model'].assert_called_once_with(cfg.model_target)
        scope['load_checkpoint'].assert_called_once_with(target, 'det.pth', map_location='cpu', strict=True)
        scope['ImageRenormTarget'].assert_called_once_with(target, CAFFE, TORCHVISION, rescale=1.0)
        self.assertIs(result, scope['ImageRenormTarget'].return_value)
        result.requires_grad_.assert_called_once_with(False)

    def test_copied_target_keeps_previous_behaviour(self):
        model = torch.nn.Linear(2, 2)
        cfg = Config(dict(load_from='seg.pth', img_norm_cfg=CAFFE))
        result, scope, _ = self.build(cfg, model)
        scope['build_model'].assert_not_called()
        scope['ImageRenormTarget'].assert_not_called()
        self.assertIsNot(result, model)
        self.assertEqual(scope['load_checkpoint'].call_args.kwargs['strict'], False)
        self.assertFalse(any(p.requires_grad for p in result.parameters()))

    def test_target_settings_without_checkpoint_fail(self):
        for extra in (dict(model_target=dict(type='BEVFormer')),
                      dict(model_target_img_norm_cfg=TORCHVISION)):
            cfg = Config(dict(load_from=None, img_norm_cfg=CAFFE, **extra))
            with self.assertRaises(FileNotFoundError):
                self.build(cfg)

    def test_rescale_without_renorm_fails(self):
        cfg = Config(dict(load_from='det.pth', img_norm_cfg=CAFFE, model_target_img_rescale=0.625))
        with self.assertRaises(ValueError):
            self.build(cfg)


if __name__ == '__main__':
    unittest.main()
