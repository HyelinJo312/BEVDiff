"""CPU checks for direct-MSE layout occupancy gradient diagnostics."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel

from projects.mmdet3d_plugin.bevformer.detectors.diff_bevformer_layout_occ import DiffBEVFormerLayoutOcc
from projects.mmdet3d_plugin.models.hooks.occ_gradient_monitor import (
    STAT_NAMES, OccGradientMonitorHook, summarize_stats,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import train_layout_occ as trainer


class TinyLayoutHead(nn.Module):
    bev_h = 2
    bev_w = 3
    embed_dims = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.norm = nn.BatchNorm2d(4)
        self.task = nn.Conv2d(4, 2, 1)
        self.last_bev = None
        self.last_map = None
        self.calls = 0
        self.encoder_calls = 0
        self.decoder_calls = 0

    def forward(self, features, metas, previous, only_bev=False, given_bev=None):
        self.calls += 1
        if given_bev is None:
            tokens = self.encoder(features)
            self.encoder_calls += 1
        else:
            tokens = given_bev
        self.last_bev = tokens
        if only_bev:
            return tokens
        bev = tokens.permute(0, 2, 1).view(1, 4, 2, 3)
        self.last_map = bev
        self.decoder_calls += 1
        return dict(bev_embed=bev, occ=self.task(self.norm(bev)))

    def loss(self, target, mask, outputs, img_metas=None):
        return dict(loss_occ=F.mse_loss(outputs['occ'], target))


class TinyLayoutTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('target', torch.full((1, 4, 2, 3), 0.2))
        self.calls = 0

    def forward(self, teacher_input, condition, grad_fn=None):
        self.calls += 1
        return self.target


def make_model():
    model = DiffBEVFormerLayoutOcc.__new__(DiffBEVFormerLayoutOcc)
    nn.Module.__init__(model)
    model.pts_bbox_head = TinyLayoutHead()
    model.distill_loss_weight = 1.0
    model.loss_mix_weight = 0.5
    model._gradient_monitor_context = None
    model._gradient_monitor_pending = None
    return model


def layout_step(model, teacher):
    condition = dict(layout_obj_classes=torch.zeros(1, 1),
                     layout_obj_bboxes=torch.zeros(1, 1, 4),
                     layout_obj_is_valid=torch.ones(1, 1))
    return model.forward_pts_train(
        torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 24,
        None, None, torch.zeros(1, 2, 2, 3), None, [{}],
        bev_diffuser=teacher, bev_target=torch.zeros(1, 6, 4),
        **condition)


class TinyDistributedLayoutStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = make_model()
        self.teacher = TinyLayoutTeacher()

    def train_step(self, data, optimizer):
        condition = dict(layout_obj_classes=torch.zeros(1, 1),
                         layout_obj_bboxes=torch.zeros(1, 1, 4),
                         layout_obj_is_valid=torch.ones(1, 1))
        losses = self.core.forward_pts_train(
            data['features'], None, None, torch.zeros(1, 2, 2, 3), None, [{}],
            bev_diffuser=self.teacher, bev_target=torch.zeros(1, 6, 4),
            **condition)
        return dict(loss=0.5 * (losses['loss_occ'] + losses['loss_bev']),
                    num_samples=1)


def distributed_layout_worker(rank, init_file, output_dir):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method='file://' + init_file,
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        model = MMDistributedDataParallel(TinyDistributedLayoutStep(),
                                          device_ids=None, broadcast_buffers=False)
        model._use_replicated_tensor_module = False
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        class Runner:
            iter = 0
            epoch = 0

        runner = Runner()
        runner.model = model.module.core
        hook = OccGradientMonitorHook(2, output_dir)
        hook.before_run(runner)
        hook.before_train_iter(runner)
        optimizer.zero_grad()
        output = model.train_step(
            dict(features=torch.ones(1, 6, 4) * (rank + 1)), optimizer)
        pending = runner.model._gradient_monitor_pending
        expected = pending.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError('Probe accumulated parameter gradients')
        output['loss'].backward()
        first_gradient = next(model.parameters()).grad
        gathered = [torch.empty_like(first_gradient) for _ in range(2)]
        dist.all_gather(gathered, first_gradient)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        optimizer.step()
        hook.after_train_iter(runner)
        if rank == 0:
            records = [json.loads(line) for line in
                       Path(hook.stream.name).read_text().splitlines()]
            for index, name in enumerate(STAT_NAMES):
                if records[1]['counts'][name] != expected[index].item():
                    raise AssertionError(f'Pooled count differs: {name}')
        hook.after_run(runner)
    finally:
        dist.destroy_process_group()


class LayoutOccGradientMonitorTests(unittest.TestCase):
    def test_split_head_matches_single_call_losses_gradients_and_buffers(self):
        torch.manual_seed(31)
        reference = make_model()
        split = copy.deepcopy(reference)
        teacher = TinyLayoutTeacher()
        features = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 24
        target = torch.zeros(1, 2, 2, 3)
        outs = reference.pts_bbox_head(features, [{}], None)
        expected = reference.pts_bbox_head.loss(target, None, outs, img_metas=[{}])
        expected['loss_bev'] = F.mse_loss(
            reference._bev_to_map(outs['bev_embed']), teacher.target,
            reduction='mean') * reference.distill_loss_weight
        actual = layout_step(split, teacher)
        self.assertEqual(reference.pts_bbox_head.calls, 1)
        self.assertEqual(split.pts_bbox_head.calls, 2)
        self.assertEqual(split.pts_bbox_head.encoder_calls, 1)
        self.assertEqual(split.pts_bbox_head.decoder_calls, 1)
        self.assertEqual(teacher.calls, 1)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(split.state_dict()[name], value, rtol=0, atol=0)
        sum(expected.values()).backward()
        sum(actual.values()).backward()
        for reference_parameter, split_parameter in zip(
                reference.parameters(), split.parameters()):
            torch.testing.assert_close(
                split_parameter.grad, reference_parameter.grad, rtol=0, atol=0)

    def test_forward_train_passes_detached_bev_target(self):
        model = make_model()
        bev_target = torch.ones(1, 6, 4, requires_grad=True)
        model_target = MagicMock(return_value=bev_target)
        teacher = TinyLayoutTeacher()
        voxel_semantics = torch.zeros(1, 2, 2, 3)
        mask_camera = torch.ones_like(voxel_semantics)
        img_metas = [[dict(prev_bev_exists=True), dict(prev_bev_exists=True)]]
        features = torch.ones(1, 6, 4)
        with patch.object(model, 'obtain_history_bev', return_value=features), \
                patch.object(model, 'extract_feat', return_value=features), \
                patch.object(model, 'forward_pts_train', return_value={}) as forward_pts:
            model.forward_train(
                img=torch.zeros(1, 2, 1, 3, 2, 2), img_metas=img_metas,
                voxel_semantics=voxel_semantics, mask_camera=mask_camera,
                model_target=model_target, bev_diffuser=teacher)
        self.assertEqual(model_target.call_count, 1)
        self.assertEqual(model_target.eval.call_count, 2)
        self.assertNotIn('clean_target', forward_pts.call_args.kwargs)
        passed_target = forward_pts.call_args.kwargs['bev_target']
        self.assertFalse(passed_target.requires_grad)
        torch.testing.assert_close(passed_target, bev_target.detach())
        self.assertIs(forward_pts.call_args.args[3], voxel_semantics)
        self.assertIs(forward_pts.call_args.args[4], mask_camera)

    def test_detection_target_returns_to_eval_after_history_and_failure(self):
        model = make_model()
        target = make_model()
        target.eval()
        def extract(**kwargs):
            target.train()
            return torch.ones(1, 6, 4)
        target.forward = extract
        image = torch.zeros(1, 2, 1, 3, 2, 2)
        metas = [[dict(prev_bev_exists=True)] * 2]
        with patch.object(model, 'obtain_history_bev', return_value=torch.ones(1, 6, 4)), \
                patch.object(model, 'extract_feat', return_value=torch.ones(1, 6, 4)), \
                patch.object(model, 'forward_pts_train', return_value={}):
            model.forward_train(img=image, img_metas=metas,
                                voxel_semantics=torch.zeros(1, 2, 2, 3),
                                model_target=target, bev_diffuser=TinyLayoutTeacher())
        self.assertFalse(target.training)
        def failing_extract(**kwargs):
            target.train()
            raise RuntimeError('history failed')
        target.forward = failing_extract
        with self.assertRaisesRegex(RuntimeError, 'history failed'):
            model.forward_train(img=image, img_metas=metas,
                                voxel_semantics=torch.zeros(1, 2, 2, 3),
                                model_target=target, bev_diffuser=TinyLayoutTeacher())
        self.assertFalse(target.training)

    def test_get_condition_preserves_tensor_and_list_inputs(self):
        for as_list in (False, True):
            with self.subTest(as_list=as_list):
                model = make_model()
                teacher = MagicMock(return_value=torch.zeros(1, 4, 2, 3))
                condition = dict(
                    layout_obj_classes=torch.zeros(1, 2),
                    layout_obj_bboxes=torch.ones(1, 2, 4),
                    layout_obj_is_valid=torch.ones(1, 2))
                kwargs = {key: list(value) if as_list else value
                          for key, value in condition.items()}
                model.forward_pts_train(
                    torch.ones(1, 6, 4), None, None,
                    torch.zeros(1, 2, 2, 3), None, [{}],
                    bev_diffuser=teacher,
                    bev_target=torch.ones(1, 6, 4, requires_grad=True), **kwargs)
                teacher_input, packed_condition = teacher.call_args.args
                self.assertEqual(tuple(teacher_input.shape), (1, 4, 2, 3))
                self.assertFalse(teacher_input.requires_grad)
                self.assertIsNone(teacher.call_args.kwargs['grad_fn'])
                for source, target in (
                        ('layout_obj_classes', 'obj_class'),
                        ('layout_obj_bboxes', 'obj_bbox'),
                        ('layout_obj_is_valid', 'is_valid_obj')):
                    torch.testing.assert_close(packed_condition[target], condition[source])

    def test_probe_preserves_loss_backward_state_and_rng(self):
        torch.manual_seed(7)
        baseline = make_model()
        monitored = copy.deepcopy(baseline)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01,
                                             momentum=0.9)
        monitored_optimizer = torch.optim.SGD(monitored.parameters(), lr=0.01,
                                              momentum=0.9)
        teacher = TinyLayoutTeacher()
        initial_keys = tuple(monitored.state_dict())
        rng = torch.random.get_rng_state()

        torch.random.set_rng_state(rng.clone())
        baseline_losses = layout_step(baseline, teacher)
        baseline_rng = torch.random.get_rng_state()
        baseline_state = copy.deepcopy(baseline.state_dict())
        baseline_loss = 0.5 * (baseline_losses['loss_occ'] + baseline_losses['loss_bev'])
        baseline_loss.backward()
        baseline_grads = [parameter.grad.clone() for parameter in baseline.parameters()]
        baseline_optimizer.step()
        baseline_updated_state = copy.deepcopy(baseline.state_dict())
        baseline_optimizer_state = copy.deepcopy(baseline_optimizer.state_dict())

        torch.random.set_rng_state(rng.clone())
        monitored._gradient_monitor_context = True
        monitored_losses = layout_step(monitored, teacher)
        self.assertFalse(monitored.pts_bbox_head.last_map.is_contiguous())
        self.assertEqual(monitored.pts_bbox_head.calls, 2)
        self.assertEqual(monitored.pts_bbox_head.encoder_calls, 1)
        self.assertEqual(monitored.pts_bbox_head.decoder_calls, 1)
        self.assertEqual(teacher.calls, 2)
        self.assertEqual(tuple(monitored.state_dict()), initial_keys)
        self.assertEqual(set(monitored_losses), {'loss_occ', 'loss_bev'})
        self.assertTrue(all(parameter.grad is None for parameter in monitored.parameters()))
        torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng)
        for key, value in monitored.state_dict().items():
            torch.testing.assert_close(value, baseline_state[key])
        for name in monitored_losses:
            torch.testing.assert_close(monitored_losses[name], baseline_losses[name])

        bev = monitored.pts_bbox_head.last_bev
        self.assertEqual(tuple(bev.shape), (1, 6, 4))
        target_tokens = teacher.target.flatten(2).transpose(1, 2)
        expected_kd = (self._kd_weight(monitored) * 2 *
                       (bev.detach() - target_tokens) / bev.numel())
        actual_kd = torch.autograd.grad(
            monitored.loss_mix_weight * monitored_losses['loss_bev'], bev,
            retain_graph=True)[0]
        torch.testing.assert_close(actual_kd, expected_kd)
        pending = monitored._gradient_monitor_pending
        self.assertEqual(tuple(pending.shape), (13,))
        self.assertFalse(pending.requires_grad)
        torch.testing.assert_close(pending[1], expected_kd.double().square().sum())
        self.assertGreater(pending[0].item(), 0)
        self.assertEqual(pending[7].item(), 0)
        self.assertEqual(pending[8].item(), 0)
        self.assertIsNone(summarize_stats(pending.tolist())['metrics']['gradmon/mgd_keep_fraction'])

        monitored_loss = 0.5 * (monitored_losses['loss_occ'] + monitored_losses['loss_bev'])
        monitored_loss.backward()
        torch.testing.assert_close(monitored_loss, baseline_loss)
        for parameter, expected in zip(monitored.parameters(), baseline_grads):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
        monitored_optimizer.step()
        for name, value in monitored.state_dict().items():
            torch.testing.assert_close(value, baseline_updated_state[name],
                                       rtol=0, atol=0)
        monitored_optimizer_state = monitored_optimizer.state_dict()
        self.assertEqual(monitored_optimizer_state['param_groups'],
                         baseline_optimizer_state['param_groups'])
        self.assertEqual(monitored_optimizer_state['state'].keys(),
                         baseline_optimizer_state['state'].keys())
        for parameter_id, state in monitored_optimizer_state['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, baseline_optimizer_state['state'][parameter_id][name],
                    rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng,
                                   rtol=0, atol=0)

    def test_train_step_cleanup_and_unscheduled_hook(self):
        model = make_model()
        model._gradient_monitor_context = True
        with patch.object(model, 'forward', return_value=dict(
                loss_occ=torch.tensor(2.0, requires_grad=True))):
            model.train_step(dict(img_metas=[{}]), None)
        self.assertIsNone(model._gradient_monitor_context)
        model._gradient_monitor_context = True
        with patch.object(model, 'forward', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                model.train_step(dict(img_metas=[{}]), None)
        self.assertIsNone(model._gradient_monitor_context)

        class Runner:
            iter = 1

        runner = Runner()
        runner.model = model
        hook = OccGradientMonitorHook(3, 'unused')
        hook.before_train_iter(runner)
        self.assertIsNone(model._gradient_monitor_context)
        self.assertIsNone(model._gradient_monitor_pending)
        hook.after_train_iter(runner)

    @staticmethod
    def _kd_weight(model):
        return model.loss_mix_weight * model.distill_loss_weight

    def test_launcher_and_cli_overlay(self):
        root = Path(__file__).resolve().parents[1]
        launcher = root / 'tools/dist_train_layout_occ.sh'
        config_path = root / 'projects/configs/diff_bevformer/layout_base_occ.py'
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / 'args.txt'
            checkpoint = Path(directory) / 'target.pth'
            checkpoint.touch()
            student_checkpoint = Path(directory) / 'student.pth'
            student_checkpoint.touch()
            unet = Path(directory) / 'stage1/unet/diffusion_pytorch_model.safetensors'
            unet.parent.mkdir(parents=True)
            unet.touch()
            script = launcher.read_text().replace(
                'UNET_CHECKPOINT_DIR="../results/bevdiffuser/baseline/BEVDiffuser_BEVFormer_base_original_bs4/checkpoint-50000"',
                f'UNET_CHECKPOINT_DIR="{unet.parent.parent}"').replace(
                'MODEL_TARGET_CHECKPOINT="./ckpts/bevformer_r101_dcn_24ep.pth"',
                f'MODEL_TARGET_CHECKPOINT="{checkpoint}"').replace(
                'LOAD_FROM="./ckpts/r101_dcn_fcos3d_pretrain.pth"',
                f'LOAD_FROM="{student_checkpoint}"')
            safe_launcher = Path(directory) / 'dist_train_layout_occ.sh'
            safe_launcher.write_text(script)
            environment = dict(os.environ, CAPTURE_ARGS=str(captured))
            wrapper = ['bash', '-c',
                       'torchrun() { printf "%s\\n" "$@" > "$CAPTURE_ARGS"; }; source "$1" "${@:2}"',
                       'launcher-test', str(safe_launcher)]
            subprocess.run(wrapper + ['--cfg-options', 'optimizer.lr=0.01',
                                      'gradient_monitor.enabled=False',
                                      'gradient_monitor.interval=10'],
                           cwd=root, env=environment, check=True)
            args = captured.read_text().splitlines()
            self.assertEqual(args[-2:], ['--gradient-monitor-enabled=True',
                                         '--gradient-monitor-interval=200'])
            self.assertIn('--nproc_per_node=8', args)
            self.assertIn('--master_port=28516', args)
            self.assertIn('--tracker_run_name=DiffBEVFormer_layout_base_occ_det_target_output_aligned', args)
            self.assertIn(f'--model-target-checkpoint={checkpoint}', args)
            self.assertIn(f'--load_from={student_checkpoint}', args)
            self.assertIn('gradient_monitor.enabled=False', args)
            with patch.object(sys, 'argv', ['train_layout_occ.py', *args[3:]]):
                parsed = trainer.parse_args()
            cfg = Config.fromfile(str(config_path))
            cfg.merge_from_dict(parsed.cfg_options)
            trainer.apply_gradient_monitor_options(cfg, parsed)
            trainer.configure_runtime(cfg, parsed)
            self.assertEqual(cfg.model_target_checkpoint, str(checkpoint))
            self.assertEqual(cfg.load_from, str(student_checkpoint))
            self.assertTrue(cfg.gradient_monitor.enabled)
            self.assertEqual(cfg.gradient_monitor.interval, 200)
            self.assertEqual(cfg.model.distill_loss_weight, 1.0)
            self.assertEqual(cfg.optimizer.lr, 0.01)
            with patch.object(sys, 'argv', ['train_layout_occ.py', 'config.py',
                                            '--gradient-monitor-interval=0']):
                invalid = trainer.parse_args()
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                trainer.apply_gradient_monitor_options(cfg, invalid)
            with patch.object(sys, 'argv', ['train_layout_occ.py', 'config.py']):
                default = trainer.parse_args()
            default_cfg = Config.fromfile(str(config_path))
            trainer.apply_gradient_monitor_options(default_cfg, default)
            self.assertFalse(default_cfg.gradient_monitor.enabled)

    def test_independent_target_checkpoint_and_hook_metadata(self):
        cfg = Config(dict(
            model=dict(type='DiffBEVFormerLayoutOcc'),
            model_target=dict(type='BEVFormer'),
            model_target_checkpoint='detection_target.pth',
            bev_diffuser_cfg=dict(unet_checkpoint_dir='stage1',
                                  noise_timesteps=5, denoise_timesteps=5,
                                  num_inference_steps=5),
            load_from='student_pretrain.pth', info_root='ckpts',
            gradient_monitor=dict(enabled=True, interval=17),
            work_dir='unused', log_level='INFO',
            data=dict(train=dict(), samples_per_gpu=1, workers_per_gpu=0,
                      shuffler_sampler=dict(), nonshuffler_sampler=dict()),
            gpu_ids=[0], seed=9, optimizer=dict(type='SGD'),
            runner=dict(type='EpochBasedRunner', max_epochs=1),
            optimizer_config=dict(), lr_config=dict(),
            checkpoint_config=dict(), log_config=dict(), workflow=[('train', 1)]))
        model = make_model()
        target_checkpoint = cfg.model_target_checkpoint
        target_model = make_model()
        with patch.object(trainer.osp, 'isfile', return_value=True), \
                patch.object(trainer, 'build_model', return_value=target_model), \
                patch.object(trainer, 'load_checkpoint'), \
                patch.object(trainer, 'BEVDiffuser', return_value=TinyLayoutTeacher()):
            target, diffuser = trainer.build_frozen_teacher(cfg)
        self.assertEqual(cfg.load_from, 'student_pretrain.pth')
        self.assertTrue(all(not parameter.requires_grad for parameter in target.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in diffuser.parameters()))
        runner = MagicMock()
        with patch.object(model, 'cuda', return_value=model), \
                patch.object(target, 'cuda', return_value=target), \
                patch.object(diffuser, 'cuda', return_value=diffuser), \
                patch.object(trainer, 'MMDataParallel', return_value=model), \
                patch.object(trainer, 'build_dataloader', return_value=[]), \
                patch.object(trainer, 'build_optimizer', return_value=object()), \
                patch.object(trainer, 'build_runner', return_value=runner), \
                patch.object(trainer, 'get_root_logger', return_value=MagicMock()):
            trainer.train_occ_model(model, [object()], cfg, validate=False,
                                    model_target=target, bev_diffuser=diffuser,
                                    target_checkpoint=target_checkpoint)
        hooks = [call.args[0] for call in runner.register_hook.call_args_list]
        self.assertEqual(len(hooks), 1)
        hook = hooks[0]
        self.assertIsInstance(hook, OccGradientMonitorHook)
        self.assertEqual(hook.interval, 17)
        self.assertEqual(hook.metadata['kind'], 'layout')
        self.assertEqual(hook.metadata['distillation_loss'], 'mse')
        self.assertEqual(hook.metadata['distill_loss_weight'], 1.0)
        self.assertEqual(hook.metadata['task_weight'], 0.5)
        self.assertEqual(hook.metadata['kd_weight'], 0.5)
        self.assertFalse(hook.metadata['use_mgd'])
        self.assertEqual(hook.metadata['teacher_checkpoint'], 'stage1')
        self.assertEqual(hook.metadata['target_checkpoint'], 'detection_target.pth')
        self.assertEqual(hook.metadata['target_config'], cfg.model_target)
        self.assertEqual(hook.metadata['teacher_transport'], 'none')
        self.assertEqual(hook.metadata['student_load_from'], cfg.load_from)
        self.assertEqual(hook.metadata['cfg_scale'], 2.0)
        self.assertEqual(hook.metadata['seed'], 9)
        self.assertEqual(hook.metadata['num_inference_steps'], 5)
        runner.load_checkpoint.assert_called_once_with(cfg.load_from)

    def test_two_rank_cpu_ddp_backward_and_hook_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.spawn(distributed_layout_worker,
                                  args=(str(Path(directory) / 'gloo-init'), directory),
                                  nprocs=2, join=True)
            output = list((Path(directory) / 'gradient_monitor').glob('*.jsonl'))
            self.assertEqual(len(output), 1)
            records = [json.loads(line) for line in output[0].read_text().splitlines()]
            self.assertEqual(records[0]['world_size'], 2)
            self.assertEqual(records[1]['iter'], 1)
            self.assertEqual(records[1]['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
