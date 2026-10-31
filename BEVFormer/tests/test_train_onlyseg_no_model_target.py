"""Check optional frozen-target construction without starting training."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mmcv import Config

BEVFORMER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BEVFORMER / 'tools'))
import train_onlyseg as trainer


class NoModelTargetTests(unittest.TestCase):
    def make_config(self, no_target=False, pretrained=True):
        return Config(dict(
            no_model_target=no_target,
            model=dict(pretrained=dict(img='torchvision://resnet50') if pretrained else None),
            bev_diffuser_cfg=dict(unet_checkpoint_dir='stage1',
                                  use_classifier_guidence=False),
            load_from='frozen_encoder.pth',
            resume_from='student_resume.pth',
            info_root='data/nuscenes',
        ))

    def test_cli_flag_and_default(self):
        for option in ([], ['--no-model-target'], ['--no_model_target']):
            with patch.object(sys, 'argv', ['train_onlyseg.py', 'config.py', *option]):
                self.assertEqual(trainer.parse_args().no_model_target, bool(option))

    def test_no_target_skips_copy_and_checkpoint_without_freezing_student(self):
        cfg = self.make_config(no_target=True)
        student, diffuser = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        with patch.object(trainer.copy, 'deepcopy') as clone, \
                patch.object(trainer, 'load_checkpoint') as load, \
                patch.object(trainer, 'BEVDiffuser', return_value=diffuser):
            target, built_diffuser = trainer.build_frozen_teacher(cfg, student)
        clone.assert_not_called()
        load.assert_not_called()
        self.assertIsNone(target)
        self.assertIs(built_diffuser, diffuser)
        self.assertIsNone(cfg.load_from)
        self.assertEqual(cfg.resume_from, 'student_resume.pth')
        self.assertTrue(all(p.requires_grad for p in student.parameters()))
        self.assertTrue(all(not p.requires_grad for p in diffuser.parameters()))

    def test_default_preserves_frozen_target_and_backbone_fallback(self):
        for pretrained in (True, False):
            cfg = self.make_config(pretrained=pretrained)
            student, diffuser = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            with patch.object(trainer, 'load_checkpoint') as load, \
                    patch.object(trainer, 'BEVDiffuser', return_value=diffuser):
                target, _ = trainer.build_frozen_teacher(cfg, student)
            self.assertIsNot(target, student)
            load.assert_called_once_with(target, 'frozen_encoder.pth', map_location='cpu')
            self.assertTrue(all(not p.requires_grad for p in target.parameters()))
            self.assertTrue(all(p.requires_grad for p in student.parameters()))
            expected = None if pretrained else 'data/nuscenes/r101_dcn_fcos3d_pretrain.pth'
            self.assertEqual(cfg.load_from, expected)

    def test_classifier_guidance_requires_target(self):
        cfg = self.make_config(no_target=True)
        cfg.bev_diffuser_cfg.use_classifier_guidence = True
        with self.assertRaisesRegex(ValueError, 'use_classifier_guidence=False'):
            trainer.build_frozen_teacher(cfg, torch.nn.Linear(2, 2))

    def test_no_diffuser_preserves_regular_student_checkpoint(self):
        cfg = self.make_config()
        cfg.bev_diffuser_cfg = None
        self.assertEqual(trainer.build_frozen_teacher(cfg, torch.nn.Linear(2, 2)),
                         (None, None))
        self.assertEqual(cfg.load_from, 'frozen_encoder.pth')

    def test_launcher_selects_v4_and_passes_flag_without_load_from(self):
        script = BEVFORMER / 'tools/dist_train_onlyseg_cache_det_v3.sh'
        command = 'torchrun() { printf "%s\\n" "$@"; }; source "$1"'
        result = subprocess.run(['bash', '-c', command, 'test', str(script)],
                                cwd=BEVFORMER, capture_output=True, text=True, check=True)
        argv = result.stdout.splitlines()
        self.assertIn('--no-model-target', argv)
        config_path = next(arg for arg in argv if arg.endswith('_stubev.py'))
        cfg = Config.fromfile(str(BEVFORMER / config_path))
        self.assertEqual(cfg.model.type, 'DiffBEVFormerSegV4')
        self.assertTrue(cfg.no_model_target)
        self.assertIsNone(cfg.load_from)
        self.assertEqual(cfg.custom_hooks, [])
        self.assertEqual(cfg.model.mgd_lambda, 0.55)
        self.assertFalse(cfg.bev_diffuser_cfg.use_classifier_guidence)
        self.assertTrue(cfg.unet.parameters.seg_bev_aligner.use_legacy_prob_embed)
        self.assertFalse(any(arg.startswith('--load_from') for arg in argv))
        checkpoint = next(arg.split('=', 1)[1] for arg in argv
                          if arg.startswith('--unet_checkpoint_dir='))
        self.assertTrue((BEVFORMER / checkpoint / 'unet/diffusion_pytorch_model.safetensors').is_file())


if __name__ == '__main__':
    unittest.main()
