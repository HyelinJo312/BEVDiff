"""Focused checks for the standalone layout-occupancy trainer."""

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mmcv import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import train_layout_occ as trainer


class LayoutOccTrainerTests(unittest.TestCase):
    def test_stage2_config_uses_shared_bases_and_explicit_task_settings(self):
        config_path = (Path(__file__).resolve().parents[1] /
                       'projects/configs/diff_bevformer/layout_base_occ.py')
        assignments = {
            target.id: statement.value
            for statement in ast.parse(config_path.read_text()).body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(ast.literal_eval(assignments['_base_']), [
            '../datasets/custom_nus-3d.py',
            '../_base_/default_runtime.py',
        ])
        self.assertTrue({
            'model', 'model_target', 'model_target_checkpoint',
            'unet', 'bev_diffuser_cfg', 'data',
            'train_pipeline', 'test_pipeline', 'evaluation',
            'optimizer', 'log_config',
        }.issubset(assignments))
        cfg = Config.fromfile(str(config_path), import_custom_modules=False)
        self.assertEqual(cfg.model.type, 'DiffBEVFormerLayoutOcc')
        self.assertEqual(cfg.model.pts_bbox_head.type, 'BEVFormerOccHead')
        detection = Config.fromfile(str(config_path.with_name('layout_base.py')),
                                    import_custom_modules=False).model
        target = cfg.model_target.copy()
        self.assertEqual(target.pop('type'), 'BEVFormer')
        detection = detection.copy()
        detection.pop('type')
        self.assertEqual(target, detection)
        self.assertEqual(cfg.custom_hooks, [])
        self.assertEqual(cfg.bev_diffuser_cfg.unet_cfg, cfg.unet)
        for split in ('train', 'val', 'test'):
            self.assertEqual(cfg.data[split].type, 'Occ3DLayoutDiffusionNuScenesDataset')
            self.assertTrue(cfg.data[split].use_layout)
        self.assertEqual(cfg.data.train.pipeline, cfg.train_pipeline)
        self.assertEqual(cfg.data.val.pipeline, cfg.test_pipeline)
        self.assertEqual(cfg.data.test.pipeline, cfg.test_pipeline)
        self.assertEqual(cfg.evaluation.pipeline, cfg.test_pipeline)

    def test_frozen_teacher_uses_independent_strict_checkpoint(self):
        cfg = Config(dict(
            model_target=dict(type='BEVFormer'),
            model_target_checkpoint='detection_bev.pth',
            bev_diffuser_cfg=dict(unet_checkpoint_dir='stage1'),
            load_from='student_pretrain.pth'))
        student = torch.nn.Linear(3, 2)
        target_model = torch.nn.Linear(3, 2)
        diffuser = torch.nn.Linear(2, 2)

        with patch.object(trainer.osp, 'isfile', return_value=True), \
                patch.object(trainer, 'build_model', return_value=target_model) as build_model, \
                patch.object(trainer, 'load_checkpoint') as load_checkpoint, \
                patch.object(trainer, 'BEVDiffuser', return_value=diffuser) as build_diffuser:
            target, built_diffuser = trainer.build_frozen_teacher(cfg)

        build_model.assert_called_once_with(cfg.model_target, train_cfg=None,
                                            test_cfg=None)
        load_checkpoint.assert_called_once_with(
            target, 'detection_bev.pth', strict=True, map_location='cpu')
        build_diffuser.assert_called_once_with(unet_checkpoint_dir='stage1')
        self.assertIsNot(target, student)
        self.assertIs(built_diffuser, diffuser)
        self.assertEqual(cfg.load_from, 'student_pretrain.pth')
        self.assertTrue(all(not param.requires_grad for param in target.parameters()))
        self.assertTrue(all(not param.requires_grad for param in diffuser.parameters()))
        self.assertFalse(target.training)
        self.assertFalse(diffuser.training)
        self.assertTrue(all(param.requires_grad for param in student.parameters()))

    def test_no_diffuser_leaves_student_checkpoint_unchanged(self):
        cfg = Config(dict(bev_diffuser_cfg=None, load_from='student.pth'))
        target, diffuser = trainer.build_frozen_teacher(cfg)

        self.assertIsNone(target)
        self.assertIsNone(diffuser)
        self.assertEqual(cfg.load_from, 'student.pth')

    def test_missing_target_checkpoint_fails_explicitly(self):
        cfg = Config(dict(model_target=dict(type='BEVFormer'),
                          model_target_checkpoint='missing.pth',
                          bev_diffuser_cfg=dict(unet_checkpoint_dir='stage1'),
                          load_from='student.pth'))
        with patch.object(trainer.osp, 'isfile', return_value=False), \
                self.assertRaisesRegex(FileNotFoundError, 'missing.pth'):
            trainer.build_frozen_teacher(cfg)
        self.assertEqual(cfg.load_from, 'student.pth')


if __name__ == '__main__':
    unittest.main()
