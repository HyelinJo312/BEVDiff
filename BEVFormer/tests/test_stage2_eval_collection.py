"""Stage 2 evaluation should not leak temporal state or drop collected results."""
import unittest
from unittest.mock import patch
import torch
from projects.mmdet3d_plugin.bevformer.apis import test as subject


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_frame_info = dict(prev_bev='old', scene_token='training', prev_pos=1, prev_angle=2)
        self.seen = []

    def forward(self, return_loss=False, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        self.seen.append(self.prev_frame_info['prev_bev'])
        self.prev_frame_info['prev_bev'] = kwargs['index']
        return [kwargs['index']]


class Loader:
    dataset = range(3)

    def __iter__(self):
        for index in self.dataset:
            yield {'index': index}


class EvaluationTests(unittest.TestCase):
    def run_evaluation(self, model):
        with patch.object(subject, 'get_dist_info', return_value=(0, 1)), \
             patch.object(subject.mmcv, 'ProgressBar'), patch.object(subject.time, 'sleep'):
            return subject.custom_multi_gpu_test(model, Loader())

    def test_repeated_evaluation_starts_fresh_and_restores_history(self):
        model = Model()
        history = model.prev_frame_info
        for _ in range(2):
            self.assertEqual(self.run_evaluation(model), [0, 1, 2])
            self.assertIs(model.prev_frame_info, history)
            self.assertTrue(model.training)
        self.assertEqual(model.seen, [None, 0, 1, None, 0, 1])
        self.assertEqual(history['prev_bev'], 'old')

    def test_failure_restores_history_and_mode(self):
        model = Model()
        history = model.prev_frame_info
        with patch.object(model, 'forward', side_effect=RuntimeError('inference failure')):
            with self.assertRaisesRegex(RuntimeError, 'inference failure'):
                self.run_evaluation(model)
        self.assertIs(model.prev_frame_info, history)
        self.assertTrue(model.training)

    def test_gpu_option_returns_collector_result(self):
        with patch.object(subject, 'collect_results_cpu', return_value=['rank0', 'rank1']):
            self.assertEqual(subject.collect_results_gpu(['rank0'], 2), ['rank0', 'rank1'])


if __name__ == '__main__':
    unittest.main()
