"""Cache evaluation must isolate state and support guidance without training grads."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mmcv import ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects/bevdiffuser'))
import test_bev_diffuser_only_seg as subject
from scheduler_utils import DDIMGuidedScheduler


class FakeUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.encoded = []
        self.forward_calls = 0

    def encode_seg(self, seg, metas, depth_maps=None, semantic_prob=None):
        assert seg is None and depth_maps is None
        assert not torch.is_grad_enabled() and not self.training
        self.encoded.append(semantic_prob.clone())
        return {1: semantic_prob}

    def forward(self, x, t, metas, seg, seg_bev_maps=None):
        assert not torch.is_grad_enabled() and not self.training
        self.forward_calls += 1
        return x * .1 + seg_bev_maps[1].mean() * self.weight


class FakeBEV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.prev_frame_info = dict(prev_bev=torch.ones(1), scene_token='training',
                                    prev_pos=1, prev_angle=2)
        self.guidance_calls = 0

    def forward(self, only_bev=False, given_bev=None, return_eval_loss=False, **kwargs):
        assert not self.training
        if only_bev:
            self.prev_frame_info['scene_token'] = 'evaluation'
            return torch.ones(1, 4, 2) * self.weight
        if return_eval_loss:
            assert isinstance(kwargs['gt_labels_3d'], subject.DataContainer)
            assert torch.is_grad_enabled()
            self.guidance_calls += 1
            return given_bev.square().mean()
        assert not torch.is_grad_enabled()
        return [{'pts_bbox': {'value': given_bev.detach()}}]


class FakeDataset:
    def __init__(self):
        self.kwargs = None

    def __len__(self):
        return 2

    def evaluate(self, results, **kwargs):
        assert len(results) == 2
        assert all(torch.isfinite(item['pts_bbox']['value']).all() for item in results)
        self.kwargs = kwargs
        return {'pts_bbox/NDS': .5}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.unet, self.bev, self.dataset = FakeUNet(), FakeBEV(), FakeDataset()
        self.scheduler = DDIMGuidedScheduler(num_train_timesteps=20, prediction_type='sample',
                                             clip_sample=False, steps_offset=1)
        self.cfg = ConfigDict(bev_h_=2, bev_w_=2, _dim_=2,
                              evaluation=dict(interval=1, by_epoch=True, dynamic_intervals=[]))
        batch = dict(img_metas=[SimpleNamespace(data=[[{}]])],
                     gt_labels_3d=[subject.DataContainer([torch.zeros(1, dtype=torch.long)])],
                     semantic_bev_probabilities=SimpleNamespace(data=[[torch.ones(3, 2, 2)]]))
        self.loader = [batch, copy.deepcopy(batch)]

    def run_eval(self, **kwargs):
        with patch.object(subject, 'get_dist_info', return_value=(0, 1)), \
             patch.object(subject, 'collect_results_cpu', side_effect=AssertionError('Unexpected distributed collective')), \
             patch.object(subject.time, 'sleep'), patch.object(subject.mmcv, 'ProgressBar'):
            return subject.evaluate(self.unet, self.bev, self.scheduler, self.dataset,
                                    self.loader, self.cfg, noise_timesteps=5,
                                    denoise_timesteps=5, num_inference_steps=2,
                                    eval=['bbox'], **kwargs)

    def test_cache_encoded_once_per_branch_and_state_restored(self):
        original_history = self.bev.prev_frame_info
        original_config = dict(self.scheduler.config)
        self.assertEqual(self.run_eval(), {'pts_bbox/NDS': .5})
        self.assertEqual(len(self.unet.encoded), 4)  # 2 batches x 2 CFG branches
        self.assertEqual(self.unet.forward_calls, 8)  # 2 batches x 2 branches x 2 steps
        for cond, uncond in zip(self.unet.encoded[::2], self.unet.encoded[1::2]):
            self.assertTrue(torch.equal(cond, torch.ones_like(cond)))
            self.assertEqual(torch.count_nonzero(uncond), 0)
        self.assertTrue(self.unet.training and self.bev.training)
        self.assertIs(self.bev.prev_frame_info, original_history)
        self.assertEqual(dict(self.scheduler.config), original_config)
        self.assertIsNone(self.scheduler.num_inference_steps)
        self.assertEqual(self.dataset.kwargs['metric'], ['bbox'])
        self.assertNotIn('by_epoch', self.dataset.kwargs)

    def test_guidance_works_inside_no_grad_without_parameter_grads(self):
        with torch.no_grad():
            self.run_eval(use_classifier_guidence=True)
        self.assertEqual(self.bev.guidance_calls, 4)
        self.assertIsNone(self.bev.weight.grad)
        self.assertIsNone(self.unet.weight.grad)

    def test_exception_restores_model_state(self):
        original_history = self.bev.prev_frame_info
        with patch.object(self.dataset, 'evaluate', side_effect=RuntimeError('metric failure')):
            with self.assertRaisesRegex(RuntimeError, 'metric failure'):
                self.run_eval()
        self.assertTrue(self.unet.training and self.bev.training)
        self.assertIs(self.bev.prev_frame_info, original_history)


if __name__ == '__main__':
    unittest.main()
