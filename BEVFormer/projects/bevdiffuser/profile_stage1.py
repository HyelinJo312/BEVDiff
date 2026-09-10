#!/usr/bin/env python
"""Opt-in torchrun shim for train_only_seg.sh; never resumes or saves checkpoints.

Set TORCHRUN_BIN to this file, PROFILE_OUTPUT to a fresh directory and PORT to
an unused port. PROFILE_STEPS defaults to 24. Timings include shared-GPU waits;
CUDA event spans are nested (projection is part of UNet), not additive.
"""

import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch():
    output = Path(os.environ['PROFILE_OUTPUT']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args = sys.argv[1:]
    entry = next(i for i, arg in enumerate(args) if arg.endswith('/train_bev_diffuser_only_seg.py'))
    training = args[entry + 1:]
    if '--resume_from_checkpoint' in training:
        raise ValueError('Profiling must not resume a production run')
    command = [sys.executable, '-m', 'torch.distributed.run', *args[:entry],
               str(Path(__file__).resolve()), '--worker', *training,
               '--output_dir', str(output), '--max_train_steps', os.environ.get('PROFILE_STEPS', '24'),
               '--seed', '0', '--checkpointing_steps', '1000000000']
    if os.environ.get('PROFILE_CONFIG'):
        command.extend(['--bev_config', os.environ['PROFILE_CONFIG']])
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    monitor = subprocess.Popen(['nvidia-smi',
        '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
        '--format=csv', '-l', '1', '--filename=' + str(output / 'gpu.csv')])
    try:
        return subprocess.call(command)
    finally:
        monitor.terminate()
        monitor.wait()


def worker():
    sys.argv.remove('--worker')
    import cv2
    import torch
    from accelerate.optimizer import AcceleratedOptimizer
    from torch.utils.data.dataloader import _BaseDataLoaderIter
    import train_bev_diffuser_only_seg as train

    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    output = Path(os.environ['PROFILE_OUTPUT']).resolve()
    rows, pending = [], {}
    events = []

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            result = original(*args, **kwargs)
            end.record()
            events.append((label, start, end, time.perf_counter() - wall))
            return result
        setattr(owner, name, measured)

    next_batch = _BaseDataLoaderIter.__next__

    def measured_next(iterator):
        if os.getpid() != main_pid:
            return next_batch(iterator)
        pending['start'] = time.perf_counter()
        result = next_batch(iterator)
        pending['data_wait_s'] = time.perf_counter() - pending['start']
        metas = result['img_metas'].data[0]
        pending['tokens'] = [meta[max(meta)]['sample_idx'] if 'sample_idx' not in meta
                             else meta['sample_idx'] for meta in metas]
        return result

    main_pid = os.getpid()
    _BaseDataLoaderIter.__next__ = measured_next
    build_bev, build_unet = train.get_bev_model, train.build_unet_v2

    def measured_bev(*args, **kwargs):
        model = build_bev(*args, **kwargs)
        wrap(model, 'forward', 'teacher')
        return model

    def measured_unet(*args, **kwargs):
        model = build_unet(*args, **kwargs)
        wrap(model.seg_aligner, 'project_semantics', 'projection')
        wrap(model, 'forward', 'unet')
        return model

    train.get_bev_model, train.build_unet_v2 = measured_bev, measured_unet
    wrap(train.Accelerator, 'backward', 'backward')
    wrap(train.Accelerator, 'gather', 'gather')
    wrap(AcceleratedOptimizer, 'step', 'optimizer')
    zero_grad = AcceleratedOptimizer.zero_grad

    def measured_zero(optimizer, *args, **kwargs):
        result = zero_grad(optimizer, *args, **kwargs)
        torch.cuda.synchronize()
        row = dict(step=len(rows) + 1, data_wait_s=pending['data_wait_s'],
                   tokens=pending['tokens'],
                   iteration_s=time.perf_counter() - pending['start'])
        for label, start, end, wall in events:
            row[label + '_gpu_s'] = row.get(label + '_gpu_s', 0) + start.elapsed_time(end) / 1000
            row[label + '_cpu_s'] = row.get(label + '_cpu_s', 0) + wall
        rows.append(row)
        events.clear()
        with (output / f'rank_{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return result

    AcceleratedOptimizer.zero_grad = measured_zero
    (output / f'environment_{rank}.json').write_text(json.dumps(dict(
        opencv_threads=cv2.getNumThreads(), torch_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), pid=main_pid), indent=2) + '\n')
    train.train()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        sys.exit(launch())
