"""Compare the cached semantic BEV (maps resized 0.5x + padded) with a full-resolution projection.

Read-only for data/semantic_bev. Run from BEVFormer/ in jhl-bevdiff on one GPU:
    CUDA_VISIBLE_DEVICES=0 python projects/bevdiffuser/analyze/compare_semantic_bev_fullres.py
For each val sample it
  1. re-projects through the cache builder's path (val pipeline geometry, 0.5x) and checks it
     reproduces the stored cache entry,
  2. projects the raw 900x1600 SAM3/Metric3D maps with the raw lidar2img (no resize/pad),
saves the full-resolution arrays, and plots cache vs full resolution next to the LiDAR/GT view.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
BEVFORMER = SCRIPT_DIR.parents[2]
sys.path[:0] = [str(BEVFORMER), str(BEVFORMER / 'projects/bevdiffuser'), str(SCRIPT_DIR)]
from mmcv import Config
from build_semantic_bev_cache import current_meta, offline_dataset
from visualize_semantic_bev_lidar_sam3 import EXTENT, lidar_panel, semantic_bev_rgb

DIFF_COLORS = {'same argmax': (0.85, 0.85, 0.85), 'different argmax': (0.85, 0.1, 0.1),
               'cache only': (0.15, 0.35, 0.9), 'full-res only': (0.1, 0.65, 0.2)}


def raw_maps(dataset, meta):
    """SAM3 ids (remapped like the cache) and Metric3D depth at the raw 900x1600 resolution."""
    seg = []
    for filename in meta['filename']:
        path = dataset._map_path(dataset.semantic_path, filename, '_mask.bin')
        raw = np.fromfile(path, dtype=np.int8).reshape(dataset.depth_raw_shape).astype(np.int16)
        remapped = raw.copy()
        for source, target in (dataset.seg_id_remap or {}).items():
            remapped[raw == source] = target
        seg.append(torch.from_numpy(remapped.astype(np.int64)))
    depth = dataset.load_depth_from_filenames(meta['filename'], None)  # no alignment: raw frame
    return torch.stack(seg), depth


def diff_map(cache, full):
    obs_c, obs_f = cache.sum(0) > 1e-6, full.sum(0) > 1e-6
    both = obs_c & obs_f
    same = both & (cache.argmax(0) == full.argmax(0))
    rgb = np.ones(cache.shape[1:] + (3,))
    rgb[same] = DIFF_COLORS['same argmax']
    rgb[both & ~same] = DIFF_COLORS['different argmax']
    rgb[obs_c & ~obs_f] = DIFF_COLORS['cache only']
    rgb[obs_f & ~obs_c] = DIFF_COLORS['full-res only']
    stats = dict(observed_cache=float(obs_c.mean()), observed_fullres=float(obs_f.mean()),
                 argmax_agreement=float(same.sum() / max(both.sum(), 1)),
                 mean_abs_diff_coobserved=float(np.abs(cache - full)[:, both].sum(0).mean()) if both.any() else 0.0,
                 cache_only_cells=int((obs_c & ~obs_f).sum()), fullres_only_cells=int((obs_f & ~obs_c).sum()))
    return rgb, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 750, 1500, 3000, 4500, 6000])
    parser.add_argument('--array-dir', default=str(BEVFORMER.parent / 'results/semantic_bev_fullres_samples'))
    parser.add_argument('--output-dir', default=str(BEVFORMER.parent / 'results/visualize/semantic_bev_fullres_compare'))
    args = parser.parse_args()
    array_dir, output = Path(args.array_dir), Path(args.output_dir)
    array_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    os.chdir(BEVFORMER / 'projects/bevdiffuser')  # config data roots are relative to this directory

    from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(str(config_path))
    dataset = offline_dataset(cfg, 'val')  # val cache entries were built through data.test.pipeline
    projector = {k: v for k, v in cfg.unet.parameters.seg_bev_aligner.items()
                 if k not in ('semantic_type', 'logcount_scale')}
    device = torch.device('cuda')
    aligner = SegBEVAligner(**projector).to(device).eval()
    cache_root = Path(cfg.semantic_bev_cache_root).resolve()
    infos = {info['token']: info for info in dataset.data_infos}

    summary = []
    for index in args.indices:
        meta = current_meta(dataset, index)
        token = meta['sample_idx']
        cache = np.load(cache_root / f'{token}.npy', allow_pickle=False)

        seg = dataset.load_segmaps(meta['filename'], meta, dataset.semantic_path)
        depth = dataset.load_depth_from_filenames(meta['filename'], meta)
        half, _ = aligner.project_semantics(seg[None].to(device), [meta], depth[None].to(device))
        half = half[0].cpu().numpy()
        reproduce = float(np.abs(half - cache).max())

        seg_raw, depth_raw = raw_maps(dataset, meta)
        meta_raw = {k: v for k, v in meta.items() if k != 'pad_shape'}
        meta_raw['lidar2img'] = meta['metric3d_lidar2img_raw']
        full, _ = aligner.project_semantics(seg_raw[None].to(device), [meta_raw], depth_raw[None].to(device))
        full = full[0].cpu().numpy()
        np.save(array_dir / f'{token}.npy', full.astype(np.float32))

        rgb_diff, stats = diff_map(cache, full)
        stats.update(index=index, token=token, cache_input_hw=list(seg.shape[-2:]),
                     reproduce_cache_max_abs=reproduce)
        summary.append(stats)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6.4))
        for ax, value, name in ((axes[0], cache, f'Cache (0.5x, {seg.shape[-2]}x{seg.shape[-1]} padded)'),
                                (axes[1], full, 'Full resolution (900x1600)')):
            rgb, mass = semantic_bev_rgb(value)
            ax.imshow(rgb, origin='lower', extent=EXTENT, interpolation='nearest')
            ax.plot(0, 0, marker='^', color='red', markersize=6)
            ax.set(title=f'{name}\nargmax, {(mass > 1e-6).mean():.1%} cells observed',
                   xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', aspect='equal')
        axes[2].imshow(rgb_diff, origin='lower', extent=EXTENT, interpolation='nearest')
        axes[2].set(title=f'Cache vs full res: argmax agreement {stats["argmax_agreement"]:.1%}\n'
                          f'mean |diff| (co-observed) {stats["mean_abs_diff_coobserved"]:.3f}',
                    xlabel='LiDAR X (m)', aspect='equal')
        axes[2].legend(handles=[Patch(color=c, label=n) for n, c in DIFF_COLORS.items()],
                       loc='upper right', fontsize=7)
        lidar_panel(axes[3], infos[token], Path.cwd())  # dataset paths are relative to the cwd set above
        fig.suptitle(f'val[{index}] token={token}   (stored cache reproduced: max |diff| {reproduce:.2e})', fontsize=10)
        path = output / f'val{index:04d}_{token}.png'
        fig.savefig(path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(json.dumps(stats))

    (output / 'summary.json').write_text(json.dumps(summary, indent=1))
    keys = ('observed_cache', 'observed_fullres', 'argmax_agreement', 'mean_abs_diff_coobserved')
    print('mean over samples:', {k: round(float(np.mean([s[k] for s in summary])), 4) for k in keys})


if __name__ == '__main__':
    main()
