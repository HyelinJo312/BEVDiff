"""Sparse, detached global BEV gradient diagnostics for Stage-2 occupancy."""

import json
import math
import os
import uuid
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from mmcv.runner import Hook


STAT_NAMES = (
    'task_squared', 'kd_squared', 'cross_dot', 'elements',
    'conflict_cells', 'valid_cells', 'cells', 'kept_cells', 'mask_cells',
    'missing_task', 'missing_kd', 'nonfinite_task', 'nonfinite_kd',
)


def gradient_sufficient_stats(task_gradient, kd_gradient, reference, mask_counts=None):
    """Return a fixed-layout detached double vector; no parameter grads are touched."""
    if reference.ndim != 4:
        raise ValueError('BEV reference must be [B,C,H,W]')
    zero = reference.new_zeros((), dtype=torch.float64)
    missing_task = task_gradient is None
    missing_kd = kd_gradient is None
    nonfinite_task = not missing_task and not bool(torch.isfinite(task_gradient).all())
    nonfinite_kd = not missing_kd and not bool(torch.isfinite(kd_gradient).all())
    valid = not (missing_task or missing_kd or nonfinite_task or nonfinite_kd)
    if not missing_task and task_gradient.shape != reference.shape:
        raise ValueError('Task gradient and BEV shape differ')
    if not missing_kd and kd_gradient.shape != reference.shape:
        raise ValueError('KD gradient and BEV shape differ')
    task_squared = (task_gradient.detach().double().square().sum()
                    if not missing_task and not nonfinite_task else zero)
    kd_squared = (kd_gradient.detach().double().square().sum()
                  if not missing_kd and not nonfinite_kd else zero)
    if valid:
        task_cells = task_gradient.detach().double()
        kd_cells = kd_gradient.detach().double()
        cell_dot = (task_cells * kd_cells).sum(dim=1)
        valid_cells = ((task_cells.square().sum(dim=1) > 0)
                       & (kd_cells.square().sum(dim=1) > 0))
        cross_dot = cell_dot.sum()
        conflict_cells = (valid_cells & (cell_dot < 0)).sum().double()
        valid_cell_count = valid_cells.sum().double()
    else:
        cross_dot = zero
        conflict_cells = zero
        valid_cell_count = zero
    kept_cells, mask_cells = mask_counts if mask_counts is not None else (zero, 0)
    values = (
        task_squared, kd_squared, cross_dot, reference.numel(),
        conflict_cells, valid_cell_count,
        reference.shape[0] * reference.shape[2] * reference.shape[3],
        kept_cells, mask_cells, int(missing_task), int(missing_kd),
        int(nonfinite_task), int(nonfinite_kd),
    )
    return torch.stack([torch.as_tensor(value, device=reference.device,
                                        dtype=torch.float64) for value in values]).detach()


def summarize_stats(values):
    """Compute pooled global statistics, leaving undefined quantities as null."""
    stats = dict(zip(STAT_NAMES, [float(value) for value in values]))
    task_squared = stats['task_squared']
    kd_squared = stats['kd_squared']
    valid = not any(stats[name] for name in (
        'missing_task', 'missing_kd', 'nonfinite_task', 'nonfinite_kd'))
    metrics = {
        'gradmon/task_l2': math.sqrt(task_squared) if valid else None,
        'gradmon/kd_l2': math.sqrt(kd_squared) if valid else None,
        'gradmon/task_rms': math.sqrt(task_squared / stats['elements'])
        if valid and stats['elements'] else None,
        'gradmon/kd_rms': math.sqrt(kd_squared / stats['elements'])
        if valid and stats['elements'] else None,
        'gradmon/kd_task_l2_ratio': math.sqrt(kd_squared / task_squared)
        if valid and task_squared > 0 else None,
        'gradmon/cosine': stats['cross_dot'] / math.sqrt(task_squared * kd_squared)
        if valid and task_squared > 0 and kd_squared > 0 else None,
        'gradmon/conflict_fraction': stats['conflict_cells'] / stats['valid_cells']
        if valid and stats['valid_cells'] else None,
        'gradmon/mgd_keep_fraction': stats['kept_cells'] / stats['mask_cells']
        if stats['mask_cells'] else None,
    }
    return dict(metrics=metrics, counts={name: stats[name] for name in STAT_NAMES},
                status='ok' if valid else 'invalid_gradient')


class OccGradientMonitorHook(Hook):
    """Probe first iteration and each interval, then all-reduce detached stats."""

    def __init__(self, interval, work_dir, metadata=None):
        if type(interval) is not int or interval < 1:
            raise ValueError('gradient_monitor.interval must be a positive integer')
        self.interval = interval
        self.work_dir = work_dir
        self.metadata = metadata or {}
        self.writer = None
        self.stream = None
        self.session = None

    @staticmethod
    def _core(runner):
        return getattr(runner.model, 'module', runner.model)

    def before_run(self, runner):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        from torch.utils.tensorboard import SummaryWriter
        output_dir = os.path.join(self.work_dir, 'gradient_monitor')
        os.makedirs(output_dir, exist_ok=True)
        self.session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex
        self.stream = open(os.path.join(output_dir, self.session + '.jsonl'), 'x')
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        header = dict(type='metadata', session=self.session, interval=self.interval,
                      start_iter=runner.iter + 1, start_epoch=runner.epoch + 1,
                      world_size=world_size, **self.metadata)
        self.stream.write(json.dumps(header) + '\n')
        self.stream.flush()
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, self.session + '.events'))

    def before_train_iter(self, runner):
        core = self._core(runner)
        core._gradient_monitor_pending = None
        core._gradient_monitor_context = (
            (runner.iter == 0 or (runner.iter + 1) % self.interval == 0))
        if not core._gradient_monitor_context:
            core._gradient_monitor_context = None

    def after_train_iter(self, runner):
        core = self._core(runner)
        pending = core._gradient_monitor_pending
        core._gradient_monitor_pending = None
        core._gradient_monitor_context = None
        due = runner.iter == 0 or (runner.iter + 1) % self.interval == 0
        if not due:
            if pending is not None:
                raise RuntimeError('Unscheduled BEV gradient probe produced statistics')
            return
        if dist.is_available() and dist.is_initialized():
            device = pending.device if pending is not None else next(core.parameters()).device
            present = torch.tensor(int(pending is not None), device=device)
            dist.all_reduce(present, op=dist.ReduceOp.MIN)
            if not bool(present):
                raise RuntimeError('Scheduled BEV gradient probe missing on at least one rank')
        elif pending is None:
            raise RuntimeError('Scheduled BEV gradient probe did not produce statistics')
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pending, op=dist.ReduceOp.SUM)
        if self.stream is not None:
            record = dict(type='probe', session=self.session, iter=runner.iter + 1,
                          epoch=runner.epoch + 1, **summarize_stats(pending.cpu().tolist()))
            self.stream.write(json.dumps(record, allow_nan=False) + '\n')
            self.stream.flush()
            for name, value in record['metrics'].items():
                if value is not None:
                    self.writer.add_scalar(name, value, runner.iter + 1)
            self.writer.flush()

    def before_val_iter(self, runner):
        core = self._core(runner)
        core._gradient_monitor_context = None
        core._gradient_monitor_pending = None

    def after_run(self, runner):
        self._core(runner)._gradient_monitor_context = None
        self._core(runner)._gradient_monitor_pending = None
        if self.writer is not None:
            self.writer.close()
        if self.stream is not None:
            self.stream.close()
