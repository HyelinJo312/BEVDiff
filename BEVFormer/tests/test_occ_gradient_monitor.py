"""Focused CPU tests for the optional occupancy BEV gradient monitor."""

import copy
import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel

from projects.mmdet3d_plugin.bevformer.detectors.diff_bevformer_occ import DiffBEVFormerOcc
from projects.mmdet3d_plugin.models.hooks.occ_gradient_monitor import (
    OccGradientMonitorHook, gradient_sufficient_stats, summarize_stats,
)


class TinyHead(nn.Module):
    bev_h = 2
    bev_w = 3
    embed_dims = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.task = nn.Linear(4, 2)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, features, metas, previous, only_bev=False, given_bev=None):
        if only_bev:
            return self.encoder(features)
        logits = self.task(given_bev)
        return {'logits': self.norm(logits.reshape(-1, 2)).reshape(1, 6, 2)}

    def loss(self, target, mask, outputs, img_metas=None):
        return {'loss_occ': F.mse_loss(outputs['logits'], target)}


class FrozenDiffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('target', torch.full((1, 4, 2, 3), 0.2))
        self.calls = 0

    def forward(self, teacher_input, metas, segmaps, depths, **kwargs):
        self.calls += 1
        return self.target


def make_model():
    model = DiffBEVFormerOcc.__new__(DiffBEVFormerOcc)
    nn.Module.__init__(model)
    model.pts_bbox_head = TinyHead()
    model.mgd_generation = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    model.use_mgd = True
    model.mgd_align = None
    model.mgd_alpha = 0.1
    model.mgd_lambda = 0.55
    model.loss_mix_weight = 0.5
    model._gradient_monitor_context = None
    model._gradient_monitor_pending = None
    return model


class TinyDistributedStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = make_model()

    def train_step(self, data, optimizer):
        losses = self.core.forward_pts_train(
            data['features'], None, None, data['target'], None, [{}],
            bev_diffuser=FrozenDiffuser(), bev_target=torch.zeros(1, 6, 4),
            semantic_bev_probabilities=torch.zeros(1, 1))
        loss, _ = self.core._parse_losses_mix(losses, weight=self.core.loss_mix_weight)
        return dict(loss=loss, num_samples=1)


def distributed_probe_worker(rank, init_file):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method='file://' + init_file,
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(23)
        model = MMDistributedDataParallel(TinyDistributedStep(), device_ids=None,
                                          broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = dict(features=torch.ones(1, 6, 4) * (rank + 1),
                    target=torch.ones(1, 6, 2))
        model.module.core._gradient_monitor_context = True
        optimizer.zero_grad()
        output = model.train_step(data, optimizer)
        pending = model.module.core._gradient_monitor_pending
        if pending is None or any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError('Probe must be detached and leave parameter.grad empty')
        output['loss'].backward()
        first_gradient = next(model.parameters()).grad
        gathered = [torch.empty_like(first_gradient) for _ in range(2)]
        dist.all_gather(gathered, first_gradient)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        optimizer.step()
        class Runner:
            iter = 0
            epoch = 0

        runner = Runner()
        runner.model = model.module.core
        if rank == 0:
            runner.model._gradient_monitor_pending = None
        hook = OccGradientMonitorHook(200, '/tmp')
        try:
            hook.after_train_iter(runner)
        except RuntimeError as error:
            if 'missing on at least one rank' not in str(error):
                raise
        else:
            raise AssertionError('Missing-rank probe must fail on both ranks')
    finally:
        dist.destroy_process_group()


class GradientMonitorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _step(self, model, enabled):
        diffuser = FrozenDiffuser()
        model._gradient_monitor_context = enabled or None
        features = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 24
        target = torch.ones(1, 6, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        losses = model.forward_pts_train(
            features, None, None, target, None, [{}], bev_diffuser=diffuser,
            bev_target=torch.zeros(1, 6, 4), semantic_bev_probabilities=torch.zeros(1, 1))
        pending = model._gradient_monitor_pending
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(diffuser.calls, 1)
        self.assertTrue(all(not parameter.requires_grad for parameter in diffuser.parameters()))
        loss, _ = model._parse_losses_mix(losses, weight=model.loss_mix_weight)
        loss.backward()
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        optimizer.step()
        return (loss.detach(), gradients, copy.deepcopy(model.state_dict()),
                torch.get_rng_state().clone(), pending)

    def test_enabled_probe_preserves_loss_gradients_update_buffers_and_rng(self):
        torch.manual_seed(17)
        baseline = make_model()
        monitored = copy.deepcopy(baseline)
        rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng.clone())
        off = self._step(baseline, False)
        torch.set_rng_state(rng.clone())
        on = self._step(monitored, True)
        torch.testing.assert_close(off[0], on[0], rtol=0, atol=0)
        for expected, actual in zip(off[1], on[1]):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for name in off[2]:
            torch.testing.assert_close(off[2][name], on[2][name], rtol=0, atol=0)
        torch.testing.assert_close(off[3], on[3], rtol=0, atol=0)
        self.assertIsNone(off[4])
        self.assertEqual(on[4].shape, (13,))
        self.assertEqual(summarize_stats(on[4].tolist())['status'], 'ok')

    def test_invalid_and_zero_gradients_are_not_fabricated(self):
        reference = torch.ones(1, 2, 2, 3)
        zero = torch.zeros_like(reference)
        missing = summarize_stats(gradient_sufficient_stats(None, zero, reference).tolist())
        self.assertIsNone(missing['metrics']['gradmon/cosine'])
        self.assertEqual(missing['counts']['missing_task'], 1)
        both_zero = summarize_stats(gradient_sufficient_stats(zero, zero, reference).tolist())
        self.assertIsNone(both_zero['metrics']['gradmon/kd_task_l2_ratio'])
        self.assertIsNone(both_zero['metrics']['gradmon/conflict_fraction'])
        bad = zero.clone()
        bad[0, 0, 0, 0] = float('nan')
        nonfinite = summarize_stats(gradient_sufficient_stats(bad, zero, reference).tolist())
        self.assertEqual(nonfinite['counts']['nonfinite_task'], 1)
        self.assertIsNone(nonfinite['metrics']['gradmon/task_l2'])

    def test_actual_mgd_mask_zeroes_kd_input_gradient(self):
        model = make_model()
        model._gradient_monitor_context = True
        captured_bev = []
        handle = model.pts_bbox_head.encoder.register_forward_hook(
            lambda module, inputs, output: captured_bev.append(output))
        random_values = torch.tensor([0.1, 0.9, 0.1, 0.9, 0.1, 0.9]).reshape(1, 1, 2, 3)
        detector_module = importlib.import_module(
            'projects.mmdet3d_plugin.bevformer.detectors.diff_bevformer_occ')
        try:
            with patch.object(detector_module.torch, 'rand', return_value=random_values):
                losses = model.forward_pts_train(
                    torch.ones(1, 6, 4), None, None, torch.ones(1, 6, 2),
                    None, [{}], bev_diffuser=FrozenDiffuser(),
                    bev_target=torch.zeros(1, 6, 4),
                    semantic_bev_probabilities=torch.zeros(1, 1))
        finally:
            handle.remove()
        kd_gradient = torch.autograd.grad(
            model.loss_mix_weight * losses['loss_bev'], captured_bev[0])[0]
        kd_map = model._bev_to_map(kd_gradient)
        torch.testing.assert_close(kd_map[:, :, random_values[0, 0] > 0.45],
                                   torch.zeros_like(kd_map[:, :, random_values[0, 0] > 0.45]),
                                   rtol=0, atol=0)
        self.assertEqual(model._gradient_monitor_pending[7].item(), 3)
        self.assertEqual(model._gradient_monitor_pending[8].item(), 6)

    def test_pooled_stats_match_concatenation_and_token_orientation(self):
        model = make_model()
        task = torch.arange(24.).reshape(1, 6, 4) - 8
        kd = -task.clone()
        task_map = model._bev_to_map(task)
        kd_map = model._bev_to_map(kd)
        self.assertEqual(task_map.shape, (1, 4, 2, 3))
        torch.testing.assert_close(task_map.flatten(2).transpose(1, 2), task)
        first = gradient_sufficient_stats(task_map, kd_map, task_map, (torch.tensor(3.), 6))
        pooled = first + first
        expected = gradient_sufficient_stats(
            torch.cat([task_map, task_map]), torch.cat([kd_map, kd_map]),
            torch.cat([task_map, task_map]), (torch.tensor(6.), 12))
        torch.testing.assert_close(pooled, expected)
        metrics = summarize_stats(pooled.tolist())['metrics']
        self.assertAlmostEqual(metrics['gradmon/cosine'], -1)
        self.assertAlmostEqual(metrics['gradmon/conflict_fraction'], 1)
        self.assertAlmostEqual(metrics['gradmon/mgd_keep_fraction'], 0.5)

    def test_schedule_resume_and_unique_files(self):
        class Runner:
            def __init__(self, model):
                self.model = model
                self.iter = 199
                self.epoch = 2

        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(make_model())
            paths = []
            for _ in range(2):
                hook = OccGradientMonitorHook(200, directory)
                hook.before_run(runner)
                hook.before_train_iter(runner)
                self.assertTrue(runner.model._gradient_monitor_context)
                runner.model._gradient_monitor_pending = gradient_sufficient_stats(
                    torch.ones(1, 2, 2, 3), torch.ones(1, 2, 2, 3),
                    torch.ones(1, 2, 2, 3))
                hook.after_train_iter(runner)
                hook.after_run(runner)
                paths.append(Path(directory) / 'gradient_monitor' / (hook.session + '.jsonl'))
            self.assertNotEqual(paths[0], paths[1])
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual(records[1]['iter'], 200)
            self.assertEqual(records[1]['epoch'], 3)
            self.assertTrue(list((Path(directory) / 'gradient_monitor').glob('*.events')))

    def test_launcher_forwards_options_and_trainer_overlays_monitor(self):
        root = Path(__file__).resolve().parents[1]
        launcher = root / 'tools/dist_train_onlyseg_cache_occ.sh'
        trainer_path = root / 'tools/train_onlyseg_occ.py'
        with patch.object(sys, 'path', [str(trainer_path.parent), *sys.path]):
            spec = importlib.util.spec_from_file_location('train_onlyseg_occ', trainer_path)
            trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(trainer)
        config_path = root / 'projects/configs/diff_bevformer/bev_onlyseg_sam_metric3d_occ.py'
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / 'args.txt'
            environment = dict(os.environ, CAPTURE_ARGS=str(captured))
            wrapper = ['bash', '-c',
                       'torchrun() { printf "%s\\n" "$@" > "$CAPTURE_ARGS"; }; source "$1" "${@:2}"',
                       'launcher-test', str(launcher)]
            command = wrapper + ['--cfg-options',
                       'optimizer.lr=0.01', 'gradient_monitor.enabled=False',
                       'gradient_monitor.interval=10', '--seed=7']
            subprocess.run(command, cwd=root, env=environment, check=True)
            args = captured.read_text().splitlines()
            self.assertEqual(args.count('--cfg-options'), 1)
            self.assertIn('optimizer.lr=0.01', args)
            self.assertIn('--seed=7', args)
            self.assertIn('gradient_monitor.enabled=False', args)
            self.assertIn('gradient_monitor.interval=10', args)
            self.assertEqual(args[-2:], ['--gradient-monitor-enabled=True',
                                         '--gradient-monitor-interval=200'])
            self.assertIn('--nproc_per_node=8', args)
            with patch.object(sys, 'argv', [str(trainer_path), *args[3:]]):
                parsed = trainer.parse_args()
            config = Config.fromfile(str(config_path))
            config.merge_from_dict(parsed.cfg_options)
            trainer.apply_gradient_monitor_options(config, parsed)
            self.assertTrue(config.gradient_monitor.enabled)
            self.assertEqual(config.gradient_monitor.interval, 200)
            self.assertEqual(config.optimizer.lr, 0.01)
            legacy = wrapper + ['--options', 'optimizer.lr=0.02']
            subprocess.run(legacy, cwd=root, env=environment, check=True)
            legacy_args = captured.read_text().splitlines()
            self.assertIn('optimizer.lr=0.02', legacy_args)
            with patch.object(sys, 'argv', [str(trainer_path), *legacy_args[3:]]):
                with self.assertWarnsRegex(UserWarning, '--options is deprecated'):
                    legacy_parsed = trainer.parse_args()
            legacy_config = Config.fromfile(str(config_path))
            legacy_config.merge_from_dict(legacy_parsed.cfg_options)
            trainer.apply_gradient_monitor_options(legacy_config, legacy_parsed)
            self.assertEqual(legacy_config.optimizer.lr, 0.02)
            self.assertTrue(legacy_config.gradient_monitor.enabled)

            with patch.object(sys, 'argv', [str(trainer_path), 'config.py',
                                            '--options', 'optimizer.lr=0.02',
                                            '--cfg-options', 'optimizer.lr=0.03']):
                with self.assertRaisesRegex(ValueError, 'cannot be both specified'):
                    trainer.parse_args()
            with patch.object(sys, 'argv', [str(trainer_path), 'config.py',
                                            '--cfg-options', '--seed=7']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    trainer.parse_args()
            self.assertEqual(raised.exception.code, 2)

            with patch.object(sys, 'argv', [str(trainer_path), 'config.py',
                                            '--gradient-monitor-enabled=False',
                                            '--gradient-monitor-interval=10']):
                disabled = trainer.parse_args()
            disabled_config = Config(dict(optimizer=dict(lr=0.01)))
            trainer.apply_gradient_monitor_options(disabled_config, disabled)
            self.assertFalse(disabled_config.gradient_monitor.enabled)
            self.assertEqual(disabled_config.gradient_monitor.interval, 10)
            self.assertEqual(disabled_config.optimizer.lr, 0.01)
            with patch.object(sys, 'argv', [str(trainer_path), 'config.py']):
                unset = trainer.parse_args()
            default_config = Config.fromfile(str(config_path))
            trainer.apply_gradient_monitor_options(default_config, unset)
            self.assertFalse(default_config.gradient_monitor.enabled)
            self.assertEqual(default_config.gradient_monitor.interval, 200)
            with patch.object(sys, 'argv', [str(trainer_path), 'config.py',
                                            '--gradient-monitor-interval=0']):
                invalid = trainer.parse_args()
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                trainer.apply_gradient_monitor_options(default_config, invalid)

    def test_interval_requires_positive_integer(self):
        for interval in (0, -1, 1.2, True, '200'):
            with self.assertRaises(ValueError):
                OccGradientMonitorHook(interval, '/tmp')

    def test_config_defaults_disabled_and_accepts_override(self):
        config_path = (Path(__file__).resolve().parents[1] /
                       'projects/configs/diff_bevformer/bev_onlyseg_sam_metric3d_occ.py')
        config = Config.fromfile(str(config_path))
        self.assertFalse(config.gradient_monitor.enabled)
        self.assertEqual(config.gradient_monitor.interval, 200)
        config.merge_from_dict({'gradient_monitor.enabled': True,
                                'gradient_monitor.interval': 10})
        self.assertTrue(config.gradient_monitor.enabled)
        self.assertEqual(config.gradient_monitor.interval, 10)

    def test_two_rank_cpu_ddp_backward_and_missing_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.spawn(distributed_probe_worker,
                                  args=(str(Path(directory) / 'gloo-init'),),
                                  nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
