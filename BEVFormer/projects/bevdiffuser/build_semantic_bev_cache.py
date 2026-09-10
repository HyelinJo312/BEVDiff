"""Build/resume local pre-embedding Metric3D caches using the real v5 projector."""

import argparse
import copy
from contextlib import contextmanager
import errno
import fcntl
from pathlib import Path
import sys

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'all'), default='all')
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--device', default='cuda', help='CUDA device matching training (e.g. cuda:0)')
    parser.add_argument('--flush-every', type=int, default=50)
    parser.add_argument('--wait-for-lock', action='store_true',
                        help='Wait for an existing writer before importing the model or initializing CUDA')
    args = parser.parse_args()
    if args.indices is not None and args.split == 'all':
        parser.error('--indices requires a single split')
    if args.flush_every < 1:
        parser.error('--flush-every must be positive')
    try:
        with cache_builder_lock(args.cache_root, wait=args.wait_for_lock):
            build_cache(args, parser)
    except CacheBuildLockedError as error:
        parser.exit(2, f'{parser.prog}: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Cache build/wait interrupted. Existing cache entries are preserved.\n')


def build_cache(args, parser):
    # Contending writers must not import the training stack or allocate GPU memory.
    import cv2
    import torch
    from mmcv import Config

    import projects.mmdet3d_plugin
    from projects.bevdiffuser.data_utils import CustomNuScenesDataset
    from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner
    from projects.bevdiffuser.semantic_bev_cache import SemanticBEVCache, cache_contract

    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        parser.error('Build on CUDA with the training environment; CPU projection is not numerically equivalent')
    cfg = Config.fromfile(args.config)
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    aligner = SegBEVAligner(**cfg.unet.parameters.seg_bev_aligner).to(args.device).eval()
    root = Path(args.cache_root)
    for split in (('train', 'val') if args.split == 'all' else (args.split,)):
        dataset = offline_dataset(cfg, split)
        contract = cache_contract(cfg.unet.parameters.seg_bev_aligner,
            dataset.depth_raw_shape, dataset.depth_range, dataset.seg_id_remap,
            cfg.semantic_bev_cache_source_version)
        cache = SemanticBEVCache(root, contract, create=True)
        cache.flush()
        indices = range(len(dataset)) if args.indices is None else args.indices
        try:
            for done, index in enumerate(indices, 1):
                # Metadata/RGB only first: resumable builds skip source maps already cached.
                item = CustomNuScenesDataset.__getitem__(dataset, index)
                meta = dataset._current_metas(item['img_metas'])
                token = meta['sample_idx']
                if token in cache.manifest['entries']:
                    cache.load(token, meta)
                    action = 'verified'
                else:
                    seg = dataset.load_segmaps(meta['filename'], meta, dataset.semantic_path)
                    depth = dataset.load_depth_from_filenames(meta['filename'], meta)
                    probability, _ = aligner.project_semantics(
                        seg.unsqueeze(0).to(args.device), [meta], depth.unsqueeze(0).to(args.device))
                    cache.save(token, meta, probability[0].cpu().numpy())
                    action = 'saved'
                if done % args.flush_every == 0:
                    cache.flush()
                print(f'{split} {done}/{len(indices)} index={index} {token}: {action}', flush=True)
        finally:
            cache.flush()
        print(f'{split}: requested samples complete; total manifest entries={len(cache.manifest["entries"])}', flush=True)


if __name__ == '__main__':
    main()
