"""Build/resume local pre-embedding Metric3D caches using the real v5 projector."""

import argparse
import copy
from contextlib import contextmanager
import errno
import fcntl
import multiprocessing
from pathlib import Path
import queue
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class CacheBuildLockedError(RuntimeError):
    pass


@contextmanager
def cache_builder_lock(root, wait=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / '.builder.lock'
    # Keep the same inode, including after release, to protect legacy builders.
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            message = (f'Another cache builder holds {lock_path}. '
                       'Only one writer per cache root is allowed, even on different GPUs. '
                       'Do not delete the lock file.')
            if not wait:
                raise CacheBuildLockedError(
                    message + ' Wait for that process to finish and rerun, '
                    'or use --wait-for-lock to queue this build.') from None
            print(message + ' Waiting for the lock; no GPU has been initialized. '
                  'Press Ctrl+C to cancel this waiting process.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print('Cache lock acquired; starting verification/build.', flush=True)
        yield


def cache_layout(cfg):
    """Stored cache layout; the config is the single source for builder and workers."""
    return cfg.get('semantic_bev_cache_layout', 'share')


def offline_dataset(cfg, split):
    import projects.mmdet3d_plugin
    from mmdet3d.datasets import build_dataset
    from projects.bevdiffuser import data_utils  # Register custom datasets for standalone callers.

    settings = copy.deepcopy(cfg.data.train if split == 'train' else cfg.data.test)
    settings.update(test_mode=True, load_annos=True, pipeline=copy.deepcopy(cfg.test_pipeline),
                    filter_empty_gt=False, use_semantic_bev_cache=False)
    settings.pop('samples_per_gpu', None)
    return build_dataset(settings, default_args=dict(pc_range=cfg.point_cloud_range,
        use_3d_bbox=cfg.use_3d_bbox, num_classes=cfg.num_classes, num_bboxes=cfg.num_bboxes))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--source-version', help='Source-map version for the build manifest; '
                        'defaults to the config value or sam3_metric3d_v1')
    parser.add_argument('--split', choices=('train', 'val', 'all'), default='all')
    parser.add_argument('--indices', nargs='+', type=int)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument('--device', default='cuda', help='Single CUDA device (e.g. cuda:0)')
    devices.add_argument('--devices', nargs='+', help='Distinct logical CUDA devices (e.g. cuda:0 cuda:1)')
    parser.add_argument('--flush-every', type=int, default=50)
    parser.add_argument('--wait-for-lock', action='store_true',
                        help='Wait for an existing writer before importing the model or initializing CUDA')
    args = parser.parse_args(argv)
    if args.indices is not None and args.split == 'all':
        parser.error('--indices requires a single split')
    if args.indices is not None and (any(index < 0 for index in args.indices)
                                    or len(set(args.indices)) != len(args.indices)):
        parser.error('--indices must be unique nonnegative dataset indices')
    if args.flush_every < 1:
        parser.error('--flush-every must be positive')
    selected = args.devices if args.devices is not None else [args.device]
    pattern = r'cuda:[0-9]+' if args.devices is not None else r'cuda(?::[0-9]+)?'
    if any(re.fullmatch(pattern, device) is None for device in selected):
        parser.error('Use --device cuda[:N] or --devices cuda:0 cuda:1 ... with logical CUDA indices')
    selected = [f'cuda:{int(device.split(":")[1])}' if ':' in device else 'cuda:0'
                for device in selected]
    if len(set(selected)) != len(selected):
        parser.error('--devices must not contain duplicate logical CUDA devices')
    args.selected_devices = selected
    return args, parser


def main():
    args, parser = parse_args()
    try:
        with cache_builder_lock(args.cache_root, wait=args.wait_for_lock):
            build_cache(args, parser)
    except CacheBuildLockedError as error:
        parser.exit(2, f'{parser.prog}: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Cache build/wait interrupted. Existing cache entries are preserved.\n')
    except Exception as error:
        parser.exit(1, f'{parser.prog}: cache build failed: {error}\n')


def current_meta(dataset, index):
    from projects.bevdiffuser.data_utils import CustomNuScenesDataset

    item = (dataset.get_unconditioned_item(index) if hasattr(dataset, 'get_unconditioned_item')
            else CustomNuScenesDataset.__getitem__(dataset, index))
    return dataset._current_metas(item['img_metas'])


def prepare_jobs(dataset, cache, indices):
    jobs = {}
    tokens = {}
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise ValueError(f'Dataset index {index} is outside [0, {len(dataset)})')
        info = dataset.data_infos[index]
        token = info.get('token', info.get('sample_token'))
        if token is None:
            raise KeyError(f'Dataset index {index} has no sample token in data_infos')
        if token in tokens:
            raise ValueError(f'Duplicate sample token {token} at indices {tokens[token]} and {index}; '
                             'refusing to overwrite its geometry')
        tokens[token] = index
        cached = token in cache.manifest['entries']
        if not cached:
            path = cache._path(token)
            if path.exists():
                raise FileExistsError(f'Orphan cache array {path} has no manifest entry; '
                                      'inspect/reconcile it manually or use a new cache root')
        jobs[index] = (token, cached)
    return jobs


def projection_worker(worker_id, config_path, split, source_version, device, tasks, results):
    try:
        import cv2
        import torch
        from mmcv import Config
        from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner
        from projects.bevdiffuser.semantic_bev_cache import cache_contract

        cfg = Config.fromfile(config_path)
        layout = cache_layout(cfg)
        if not torch.cuda.is_available() or torch.device(device).index is not None and torch.device(device).index >= torch.cuda.device_count():
            raise ValueError(f'{device} is not an available logical CUDA device')
        torch.cuda.set_device(device)
        if cfg.get('close_tf32', False):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        cv2.setNumThreads(1)
        dataset = offline_dataset(cfg, split)
        contract = cache_contract(cfg.unet.parameters.seg_bev_aligner,
            dataset.depth_raw_shape, dataset.depth_range, dataset.seg_id_remap, source_version,
            layout=layout)
        # Projection is independent of the model-input encoding (semantic_type).
        projector = {key: value for key, value in cfg.unet.parameters.seg_bev_aligner.items()
                     if key not in ('semantic_type', 'logcount_scale')}
        aligner = SegBEVAligner(**projector).to(device).eval()
        results.put(('ready', worker_id, contract, torch.cuda.get_device_name(device)))
        while True:
            task = tasks.get()
            if task is None:
                return
            index, expected_token, cached = task
            meta = current_meta(dataset, index)
            if meta['sample_idx'] != expected_token:
                raise ValueError(f'Index {index} returned token {meta["sample_idx"]}, expected {expected_token}')
            if cached:
                value = None
            else:
                seg = dataset.load_segmaps(meta['filename'], meta, dataset.semantic_path)
                depth = dataset.load_depth_from_filenames(meta['filename'], meta)
                if layout == 'share':
                    probability, _ = aligner.project_semantics(
                        seg.unsqueeze(0).to(device), [meta], depth.unsqueeze(0).to(device))
                    value = probability[0].cpu().numpy()
                else:
                    probability, support, counts = aligner.project_semantics(
                        seg.unsqueeze(0).to(device), [meta], depth.unsqueeze(0).to(device),
                        return_counts=True)
                    value = torch.cat([probability[0], counts[0], support[0]], dim=0).cpu().numpy()
            results.put(('result', worker_id, index, meta, value))
    except Exception:
        results.put(('error', worker_id, traceback.format_exc()))


def _receive(results, processes):
    try:
        message = results.get_nowait()
    except queue.Empty:
        failed = [(number, process.exitcode) for number, process in enumerate(processes)
                  if process.exitcode is not None]
        if failed:
            raise RuntimeError(f'Cache worker exited without a result: {failed}')
        try:
            message = results.get(timeout=1)
        except queue.Empty:
            failed = [(number, process.exitcode) for number, process in enumerate(processes)
                      if process.exitcode is not None]
            if failed:
                raise RuntimeError(f'Cache worker exited without a result: {failed}')
            return None
    if message[0] == 'error':
        raise RuntimeError(f'Cache worker {message[1]} failed:\n{message[2]}')
    if message[0] == 'result':
        failed = [(number, process.exitcode) for number, process in enumerate(processes)
                  if process.exitcode is not None]
        if failed:
            raise RuntimeError(f'Cache worker exited before shutdown: {failed}')
    return message


def run_workers(config_path, split, source_version, devices, jobs, cache, flush_every,
                worker_target=projection_worker):
    if not jobs:
        return
    devices = devices[:min(len(devices), len(jobs))]
    context = multiprocessing.get_context('spawn')
    results = context.Queue(maxsize=max(2, 2 * len(devices)))
    task_queues = [context.Queue(maxsize=1) for _ in devices]
    processes = [context.Process(target=worker_target,
        args=(number, config_path, split, source_version, device, task_queues[number], results))
        for number, device in enumerate(devices)]
    try:
        for process in processes:
            process.start()
        ready = set()
        while len(ready) < len(processes):
            message = _receive(results, processes)
            if message is None:
                continue
            if message[0] != 'ready' or message[1] in ready or message[2] != cache.contract:
                raise RuntimeError(f'Worker projection/runtime contract mismatch or invalid startup: {message[:2]}')
            ready.add(message[1])
            print(f'worker {message[1]} device={devices[message[1]]} GPU={message[3]} ready', flush=True)
        remaining = iter(jobs)
        active = {}
        for number, tasks in enumerate(task_queues):
            index = next(remaining, None)
            if index is not None:
                tasks.put((index, *jobs[index]))
                active[number] = index
        completed = 0
        saved = 0
        while active:
            message = _receive(results, processes)
            if message is None:
                continue
            if message[0] != 'result' or active.get(message[1]) != message[2]:
                raise RuntimeError(f'Unexpected or duplicate cache worker result: {message[:3]}')
            _, number, index, meta, value = message
            token, cached = jobs[index]
            if meta['sample_idx'] != token:
                raise ValueError(f'Worker returned wrong token for index {index}')
            if cached:
                if value is not None:
                    raise ValueError(f'Worker returned a new array for existing token {token}')
                cache.load(token, meta)
                action = 'verified'
            else:
                if value is None:
                    raise ValueError(f'Worker omitted projection for new token {token}')
                if token in cache.manifest['entries'] or cache._path(token).exists():
                    raise FileExistsError(f'Cache entry or orphan array appeared during build: {token}')
                cache.save(token, meta, value)
                saved += 1
                if saved % flush_every == 0:
                    cache.flush()
                action = 'saved'
            completed += 1
            print(f'{split} {completed}/{len(jobs)} index={index} {token}: {action}', flush=True)
            del active[number]
            next_index = next(remaining, None)
            if next_index is not None:
                task_queues[number].put((next_index, *jobs[next_index]))
                active[number] = next_index
        for tasks in task_queues:
            tasks.put(None)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f'Cache worker exited with code {process.exitcode}')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join()
        for tasks in task_queues:
            tasks.close()
        results.close()


def build_cache(args, parser):
    import torch
    from mmcv import Config
    from projects.bevdiffuser.semantic_bev_cache import SemanticBEVCache, cache_contract

    if not torch.cuda.is_available() or any(
            torch.device(device).index is not None and torch.device(device).index >= torch.cuda.device_count()
            for device in args.selected_devices):
        parser.error('Selected logical CUDA device is unavailable; CPU projection is not numerically equivalent')
    cfg = Config.fromfile(args.config)
    source_version = args.source_version or cfg.get('semantic_bev_cache_source_version', 'sam3_metric3d_v1')
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root = Path(args.cache_root)
    for split in (('train', 'val') if args.split == 'all' else (args.split,)):
        dataset = offline_dataset(cfg, split)
        contract = cache_contract(cfg.unet.parameters.seg_bev_aligner,
            dataset.depth_raw_shape, dataset.depth_range, dataset.seg_id_remap,
            source_version, layout=cache_layout(cfg))
        cache = SemanticBEVCache(root, contract, create=True)
        print(f'{split}: cache layout={cache.layout} stored shape={cache.shape}', flush=True)
        indices = range(len(dataset)) if args.indices is None else args.indices
        try:
            jobs = prepare_jobs(dataset, cache, indices)
            run_workers(args.config, split, source_version, args.selected_devices, jobs, cache,
                        args.flush_every)
        finally:
            cache.flush()
        print(f'{split}: requested samples complete; total manifest entries={len(cache.manifest["entries"])}', flush=True)


if __name__ == '__main__':
    main()
