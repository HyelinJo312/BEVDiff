"""Inspect cached semantic BEV tensors before Stage 1 CFG dropout/embedding."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mmcv import Config
from mmcv.parallel import collate

SCRIPT_DIR = Path(__file__).resolve().parent
BEVFORMER = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(BEVFORMER))
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
from mmdet.apis import set_random_seed
from projects.bevdiffuser.data_utils import CustomNuScenesDiffusionDataset_seg_depth_v2 as Dataset
from projects.bevdiffuser.semantic_bev_cache import SemanticBEVCache, geometry_digest
from BEVFormer.projects.bevdiffuser.analyze.debug_seg_bev import CLASS_COLORS, SEG_CLASS_NAMES


def plot_map(ax, value, extent, title, **kwargs):
    artist = ax.imshow(value, origin='lower', extent=extent, interpolation='nearest', **kwargs)
    ax.plot(0, 0, '+', color='white', markersize=6)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', aspect='equal')
    return artist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 1000, 10000, 20000])
    parser.add_argument('--output-dir', default=str(BEVFORMER.parent / 'results/visualize/semantic_bev_cache'))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    os.chdir(SCRIPT_DIR)  # Same relative data roots as the training launcher.
    output.mkdir(parents=True, exist_ok=True)
    set_random_seed(0, deterministic=False)
    cfg = Config.fromfile(str(config_path))
    settings = copy.deepcopy(cfg.data.train)
    if not settings.get('use_semantic_bev_cache', False):
        raise ValueError('Select a config with use_semantic_bev_cache=True')
    dataset = build_dataset(settings, default_args=dict(
        pc_range=cfg.point_cloud_range, use_3d_bbox=cfg.use_3d_bbox,
        num_classes=cfg.num_classes, num_bboxes=cfg.num_bboxes))
    # Retain one extra metadata field for the legacy geometry check only.
    # This does not change image transforms or the cached probability tensor.
    if not dataset._retain_raw_projection(dataset.pipeline):
        raise ValueError('Cannot retain raw projection metadata for the legacy comparison')
    root = Path(dataset.semantic_bev_cache_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    # Forensic comparison under the STORED contract, not certification that the
    # current runtime/code would reproduce the original projection.
    legacy = SemanticBEVCache(root, manifest['contract'])
    infos = {info['token']: info for info in dataset.data_infos}
    samples, records, clouds, legacy_samples = [], [], [], []
    with patch.object(Dataset, 'load_segmaps', side_effect=AssertionError('Raw semantic access')), \
            patch.object(Dataset, 'load_depth_from_filenames', side_effect=AssertionError('Raw depth access')):
        for index in args.indices:
            item = dataset[index]
            meta = dataset._current_metas(item['img_metas'])
            # Same MMCV collation and torch.stack as train_bev_diffuser_only_seg.py.
            batch = collate([{'semantic_bev_probabilities': item['semantic_bev_probabilities']}], samples_per_gpu=1)
            probability = torch.stack(batch['semantic_bev_probabilities'].data[0], dim=0)[0].cpu().numpy()
            cache_path = Path(dataset.semantic_bev_cache_root) / (meta['sample_idx'] + '.npy')
            direct = np.load(cache_path, allow_pickle=False)
            np.testing.assert_array_equal(probability, direct)
            # Do not suppress legacy validation failures or label the result as a
            # successful legacy load. Compare file contents separately in that case.
            legacy_error = None
            try:
                old_probability = legacy.load(meta['sample_idx'], meta)
            except ValueError as error:
                legacy_error = str(error)
                if 'geometry mismatch' not in legacy_error:
                    raise
                old_probability = np.load(cache_path, allow_pickle=False)
                legacy._validate(old_probability)
                print('Legacy check:', legacy_error, flush=True)
            entry = manifest['entries'][meta['sample_idx']]
            checksum = hashlib.sha256(old_probability.tobytes()).hexdigest()
            if checksum != entry['sha256']:
                raise ValueError(f'Cache checksum mismatch: {cache_path}')
            np.testing.assert_array_equal(probability, old_probability)
            if probability.dtype != old_probability.dtype or probability.tobytes() != old_probability.tobytes():
                raise AssertionError('Cached tensor dtype or bytes changed')
            lidar_path = Path(infos[meta['sample_idx']]['lidar_path'])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            lo, hi = np.asarray(cfg.point_cloud_range[:3]), np.asarray(cfg.point_cloud_range[3:])
            visible = np.isfinite(points).all(1) & (points >= lo).all(1) & (points < hi).all(1)
            clouds.append(points[visible])
            legacy_samples.append(old_probability)
            if not np.isfinite(probability).all() or (probability < 0).any():
                raise ValueError(f'Nonfinite or negative cache values: {cache_path}')
            mass = probability.sum(0)
            valid = mass > 0
            share = np.divide(probability.max(0), mass, out=np.zeros_like(mass), where=valid)
            records.append(dict(requested_dataset_index=index, token=meta['sample_idx'],
                cache_path=str(cache_path.resolve()), shape=list(probability.shape), dtype=str(probability.dtype),
                lidar_path=str(lidar_path.resolve()), lidar_points_total=len(points),
                lidar_points_in_pc_range=int(visible.sum()),
                legacy_comparison=dict(stored_contract_used=True, legacy_load_succeeded=legacy_error is None,
                    legacy_error=legacy_error, geometry_check_passed=legacy_error is None,
                    stored_geometry=entry['geometry'], current_geometry=geometry_digest(meta),
                    checksum_check_passed=True, dtype_equal=True, bytes_equal=True,
                    different_elements=int(np.count_nonzero(probability != old_probability)),
                    max_absolute_difference=float(np.abs(probability - old_probability).max()),
                    sha256=hashlib.sha256(probability.tobytes()).hexdigest()),
                minimum=float(probability.min()), maximum=float(probability.max()),
                total_mass_min=float(mass.min()), total_mass_max=float(mass.max()),
                observed_fraction=float(valid.mean()), mean_dominant_share_observed=float(share[valid].mean()) if valid.any() else None,
                mean_mass_per_class=probability.mean((1, 2)).tolist()))
            samples.append(probability)
            print('Loaded', index, meta['sample_idx'], probability.shape, flush=True)

    colors = np.asarray(CLASS_COLORS, dtype=np.float32)[:, :3]
    unknown = np.array([.15, .15, .15])
    names = dict(SEG_CLASS_NAMES)
    names[0] = 'unknown (unused channel)'
    names[2] = 'highway / road'
    extent = [cfg.point_cloud_range[i] for i in (0, 3, 1, 4)]
    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 5.3 * len(samples)), squeeze=False)
    for row, (prob, record) in enumerate(zip(samples, records)):
        mass = prob.sum(0)
        valid = mass > 0
        dominant = colors[prob.argmax(0)].copy()
        dominant[~valid] = unknown
        label = f"index {record['requested_dataset_index']} | {record['token'][:10]}"
        cloud = clouds[row]
        ax = axes[row, 0]
        ax.set_facecolor('#181818')
        scatter = ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], s=.4,
                             cmap='viridis', vmin=cfg.point_cloud_range[2], vmax=cfg.point_cloud_range[5])
        ax.plot(0, 0, '+', color='white', markersize=6)
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal',
               xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', title=label + '\nLIDAR_TOP (height-colored)')
        fig.colorbar(scatter, ax=ax, fraction=.046, pad=.04, label='z (m)')
        old = legacy_samples[row]
        old_rgb = colors[old.argmax(0)].copy()
        old_rgb[old.sum(0) == 0] = unknown
        plot_map(axes[row, 1], old_rgb, extent, 'Stored .npy (original checksum verified)')
        plot_map(axes[row, 2], dominant, extent, 'Current np.load training tensor\nAll channels: exact byte match')
        plot_map(axes[row, 3], dominant, extent, 'Current semantic BEV + LIDAR_TOP')
        axes[row, 3].scatter(cloud[:, 0], cloud[:, 1], color='white', s=.35, alpha=.5)
        channel_fig, channel_axes = plt.subplots(3, 6, figsize=(19, 10), constrained_layout=True)
        for channel, ax in enumerate(channel_axes.flat):
            if channel >= len(prob):
                ax.axis('off')
                continue
            im = plot_map(ax, prob[channel], extent, f'{channel}: {names[channel]}', cmap='magma', vmin=0, vmax=1)
        channel_fig.colorbar(im, ax=list(channel_axes.flat), shrink=.65, label='Cached channel mass (no renormalization)')
        channel_fig.suptitle(f"Cache channels | index {record['requested_dataset_index']} | {record['token']}")
        filename = f"channels_{record['requested_dataset_index']:05d}_{record['token']}.png"
        channel_fig.savefig(output / filename, dpi=150)
        plt.close(channel_fig)
        record['channels_figure'] = filename
    legend = [Patch(facecolor=unknown, label='Unobserved (all channels zero)')]
    legend.extend(Patch(facecolor=colors[i], label=f'{i}: {names[i]}') for i in range(1, 17))
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=9, frameon=False)
    fig.suptitle('Semantic BEV cache comparison + same-sample LIDAR_TOP | Before CFG dropout / embedding', fontsize=15)
    fig.tight_layout(rect=(0, .075, 1, .97), h_pad=3, w_pad=2)
    fig.savefig(output / 'overview.png', dpi=160)
    plt.close(fig)
    (output / 'statistics.json').write_text(json.dumps(dict(config=str(config_path), comparison_scope='Stored file + original checksum vs current dataset + collation; legacy validation attempted separately; no reprojection', stored_contract=manifest['contract'], samples=records), indent=2))
    (output / 'README.md').write_text('''# Semantic BEV cache inspection with LIDAR_TOP

- overview.png: same-sample LIDAR_TOP, stored .npy semantic map (checksum verified),
  current np.load training tensor semantic map, and current map with LiDAR overlay.
- channels_*.png: complete current 17-channel arrays, with no renormalization.
- statistics.json: source paths, comparisons, validation outcomes and stored contract.

Actual configured training Dataset tensors are collated/stacked as in training,
BEFORE CFG dropout and learned embedding. Raw semantic/depth loaders are disabled.
Selected dataset indices are spread across training, not a shuffled training batch.
LIDAR_TOP is selected by the actual returned sample token, in its native LiDAR
frame. Points are cropped to the config's XYZ point_cloud_range. No point sampling.
Horizontal = LiDAR X; vertical = LiDAR Y; white cross = origin.
Gray means all channels zero, not confirmed free space. Argmax is only used for
plotting: training receives all 17 floating-point channels.

The legacy SemanticBEVCache.load is attempted using the contract STORED in the
manifest. Per-sample results/errors are explicitly recorded. If its geometry
check fails, the comparison reference is the .npy file, validated with the legacy
shape/dtype/finite/nonnegative checks and original manifest SHA256. This fallback
is NOT a successful legacy load. Exact full-array/dtype/byte comparisons against
the current Dataset + MMCV collate + torch.stack path must pass in either case.
The legacy reader itself uses np.load and returns that array without alteration.
Direct loading removes metadata/integrity checks, not semantic channels.

This verifies saved-array loading fidelity for the selected samples, not all
cached samples, geometry equivalence or fresh semantic/depth reprojection.
It does NOT certify current code/runtime matches the stored build contract.
A geometry-hash mismatch cannot identify which geometry field differs because
the manifest stores only its hash, not the original matrices/shapes.
Apparent semantic/LiDAR misalignment can originate in depth/semantic estimation,
projection, visibility or BEV rasterization. It is not evidence of np.load loss;
use the exact array comparisons to distinguish the two issues.
''')
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
