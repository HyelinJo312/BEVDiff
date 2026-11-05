"""Semantic BEV (detection) with vs without seg_id_remap, plus a nuScenes-style LIDAR_TOP.

All maps are projected live with the current v5 projector from the same SAM3 masks
and Metric3D depth (cache builder data path), so only the label handling differs:

  remap            : config as used for training. seg_id_remap={16:2,17:2,18:2,19:16},
                     num_classes=16 (crosswalk/lane/arrow -> road, sky -> 16 ignored).
  no remap (cfg)   : same projector, seg_id_remap removed. Raw 16 (crosswalk) equals the
                     sky id and is dropped; raw 17-19 exceed num_classes and are dropped.
  no remap (raw)   : seg_id_remap removed and num_classes=19, so crosswalk/lane/arrow
                     stay separate classes and raw 19 (sky) is the ignored id.

LIDAR_TOP follows nuscenes-devkit render_sample_data: white background, points
coloured by horizontal distance (viridis), GT boxes in the official category colours.
Everything is in the LiDAR frame: horizontal = LiDAR X, vertical = LiDAR Y.

Run from BEVFormer/projects/bevdiffuser:
  CUDA_VISIBLE_DEVICES=<gpu> python analyze/visualize_semantic_bev_remap.py
"""
import argparse
import copy
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
from mmcv import Config

SCRIPT_DIR = Path(__file__).resolve().parent
BEVDIFFUSER = SCRIPT_DIR.parent
BEVFORMER = BEVDIFFUSER.parents[1]
sys.path.insert(0, str(BEVFORMER))
sys.path.insert(0, str(BEVDIFFUSER))
from build_semantic_bev_cache import offline_dataset, current_meta
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner

# Model taxonomy after remap (0-16) and raw SAM3 extras (16-19), SAM3 palette.
NAMES = {0: 'unknown', 1: 'car', 2: 'road', 3: 'bus', 4: 'truck', 5: 'terrain', 6: 'tree',
         7: 'sidewalk', 8: 'bicycle', 9: 'barrier', 10: 'pedestrian', 11: 'manmade',
         12: 'motorcycle', 13: 'construction veh.', 14: 'trailer', 15: 'cone'}
RAW_EXTRA = {16: 'crosswalk', 17: 'lane marking', 18: 'road arrow', 19: 'sky'}
PALETTE = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    [0, 175, 0], [255, 0, 0], [255, 20, 147], [135, 60, 0], [160, 32, 240], [255, 0, 255],
    [139, 137, 137], [75, 0, 75], [150, 240, 80], [184, 134, 11],
    [255, 255, 255], [0, 255, 127], [255, 165, 0], [135, 206, 235]], dtype=np.float32) / 255
UNOBSERVED = np.array([.15, .15, .15])


VARIANT_TITLES = dict(
    remap='with seg_id_remap (training config)\ncrosswalk/lane/arrow -> road',
    cfg='no remap, config as-is (num_classes=16)\ncrosswalk dropped as sky id; lane/arrow dropped',
    raw='no remap, raw classes kept (num_classes=19)\ncrosswalk/lane/arrow separate')
CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def raw_sam3(ds, filename):
    path = ds._map_path(ds.semantic_path, filename, '_mask.bin')
    return np.fromfile(path, dtype=np.int8).reshape(ds.depth_raw_shape).astype(np.int16)


def save_sample_figure(output, ds, meta, nusc, index, split, extent, probs):
    """6 cameras with raw SAM3 labels (no remap) overlaid, then LiDAR + semantic BEVs."""
    from PIL import Image
    by_cam = {next(c for c in CAMERAS if f'/{c}/' in f): f for f in meta['filename']}
    ncol = max(3, 1 + len(probs))
    fig = plt.figure(figsize=(6.4 * ncol, 6.4 + 2 * 3.1 * ncol / 3))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1, 1, 2.1 * 3 / ncol])
    cam_cols = np.array_split(np.arange(ncol), 3)
    for k, cam in enumerate(CAMERAS):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(gs[r, cam_cols[c][0]:cam_cols[c][-1] + 1])
        image = np.asarray(Image.open(by_cam[cam]).convert('RGB'), dtype=np.float32) / 255
        labels = raw_sam3(ds, by_cam[cam])
        valid = (labels > 0) & (labels < len(PALETTE))
        overlay = image.copy()
        overlay[valid] = 0.45 * image[valid] + 0.55 * PALETTE[labels[valid]]
        ax.imshow(overlay[::2, ::2])
        ax.set_title(cam, fontsize=11)
        ax.axis('off')
    ax = fig.add_subplot(gs[2, 0])
    n = plot_lidar_nuscenes(ax, nusc, meta['sample_idx'], extent)
    ax.set_title(f'LIDAR_TOP + GT boxes ({n})')
    for c, (name, prob) in enumerate(probs.items(), start=1):
        plot_bev(fig.add_subplot(gs[2, c]), dominant_rgb(prob), extent, VARIANT_TITLES[name])
    legend = [Patch(facecolor=UNOBSERVED, label='unobserved (BEV)')]
    legend += [Patch(facecolor=PALETTE[i], edgecolor='k', label=f'{i}: {n}') for i, n in NAMES.items() if i > 0]
    legend += [Patch(facecolor=PALETTE[i], edgecolor='k', label=f'raw {i}: {n}') for i, n in RAW_EXTRA.items()]
    fig.legend(handles=legend, loc='lower center', ncol=10, fontsize=10, frameon=False)
    fig.suptitle(f'{split} {index} | {meta["sample_idx"]} | cameras: raw SAM3 labels (no remap) | '
                 'bottom: LiDAR and semantic BEV (LiDAR frame)', fontsize=14)
    fig.tight_layout(rect=(0, .05, 1, .97))
    fig.savefig(output / f'sample_{index:05d}_{meta["sample_idx"]}.png', dpi=100)
    plt.close(fig)


def dominant_rgb(prob):
    """Argmax colour per cell over real classes; gray where no class mass."""
    classes = prob[1:]
    rgb = PALETTE[1 + classes.argmax(0)].copy()
    rgb[classes.sum(0) <= 0] = UNOBSERVED
    return rgb


def plot_bev(ax, rgb, extent, title):
    ax.imshow(rgb, origin='lower', extent=extent, interpolation='nearest')
    ax.plot(0, 0, '+', color='white', markersize=7)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', aspect='equal')


def plot_lidar_nuscenes(ax, nusc, sample_token, extent):
    """nuscenes-devkit render_sample_data style for LIDAR_TOP (single sweep)."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import BoxVisibility
    lidar_token = nusc.get('sample', sample_token)['data']['LIDAR_TOP']
    path, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.ANY)
    points = LidarPointCloud.from_file(path).points
    axes_limit = max(abs(v) for v in extent)
    dists = np.sqrt(np.sum(points[:2] ** 2, axis=0))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
    ax.set_facecolor('white')
    ax.scatter(points[0], points[1], c=colors, s=0.2, cmap='viridis', rasterized=True)
    ax.plot(0, 0, 'x', color='red')                                # ego (devkit marker)
    for box in boxes:
        c = np.array(nusc.colormap[box.name]) / 255.0
        box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1.0)
    ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal', xlabel='LiDAR X (m)',
           ylabel='LiDAR Y (m)', title='LIDAR_TOP + GT boxes')
    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 1000, 10000, 20000])
    parser.add_argument('--num-samples', type=int, default=None,
                        help='If set, use this many indices spread evenly over the split (overrides --indices).')
    parser.add_argument('--variants', nargs='+', default=['remap', 'cfg', 'raw'], choices=('remap', 'cfg', 'raw'),
                        help="BEV columns: remap = with seg_id_remap, cfg = no remap (num_classes=16), "
                             "raw = no remap with raw classes kept (num_classes=19)")
    parser.add_argument('--with-seg', action='store_true',
                        help='Also save per-sample figures: 6 cameras with SAM3 overlay + LiDAR + BEVs.')
    parser.add_argument('--rows-per-page', type=int, default=4)
    parser.add_argument('--split', default='train', choices=('train', 'val'))
    parser.add_argument('--output-dir', default=str(BEVFORMER.parent / 'results/visualize/semantic_bev_remap'))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(BEVDIFFUSER)                       # same relative data roots as training
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = Config.fromfile(args.config)
    remap = dict(cfg.data.train.get('seg_id_remap') or {})
    ds = offline_dataset(cfg, args.split)
    if args.num_samples:
        args.indices = np.linspace(0, len(ds) - 1, args.num_samples).round().astype(int).tolist()
    projector = {k: v for k, v in cfg.unet.parameters.seg_bev_aligner.items()
                 if k not in ('semantic_type', 'logcount_scale')}
    dev = 'cuda'
    al16 = SegBEVAligner(**projector).to(dev).eval()
    al19 = SegBEVAligner(**{**projector, 'num_classes': 19}).to(dev).eval()
    extent = [cfg.point_cloud_range[i] for i in (0, 3, 1, 4)]
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(Path(ds.data_root).resolve()), verbose=False)

    rows, stats = [], []
    for index in args.indices:
        meta = current_meta(ds, index)
        depth = ds.load_depth_from_filenames(meta['filename'], meta).unsqueeze(0).to(dev)
        ds.seg_id_remap = remap
        seg_remap = ds.load_segmaps(meta['filename'], meta, ds.semantic_path).unsqueeze(0).to(dev)
        ds.seg_id_remap = None
        seg_raw = ds.load_segmaps(meta['filename'], meta, ds.semantic_path).unsqueeze(0).to(dev)
        with torch.no_grad():
            p_remap, s_remap = al16.project_semantics(seg_remap, [meta], depth)
            p_cfg, s_cfg = al16.project_semantics(seg_raw, [meta], depth)
            p_raw, s_raw = al19.project_semantics(seg_raw, [meta], depth)
        p_remap, p_cfg, p_raw = (x[0].cpu().numpy() for x in (p_remap, p_cfg, p_raw))
        support = {k: float(s[0].sum()) for k, s in (('remap', s_remap), ('no_remap_cfg', s_cfg), ('no_remap_raw', s_raw))}
        raw_px = {RAW_EXTRA[i]: int((seg_raw == i).sum()) for i in RAW_EXTRA}
        observed = lambda p: (p[1:].sum(0) > 0)
        changed = observed(p_remap) & observed(p_cfg) & (p_remap[1:16].argmax(0) != p_cfg[1:16].argmax(0))
        rec = dict(index=index, token=meta['sample_idx'], raw_sam3_pixels=raw_px,
                   projected_pixel_mass=support,
                   observed_cells=dict(remap=int(observed(p_remap).sum()), no_remap_cfg=int(observed(p_cfg).sum()),
                                       no_remap_raw=int(observed(p_raw).sum())),
                   road_share_mean_on_observed=dict(remap=float(p_remap[2][observed(p_remap)].mean()),
                                                    no_remap_cfg=float(p_cfg[2][observed(p_cfg)].mean())),
                   cells_dominant_class_changed_remap_vs_no_remap_cfg=int(changed.sum()),
                   raw_marking_share_mean_on_observed={RAW_EXTRA[c]: float(p_raw[c][observed(p_raw)].mean()) for c in (16, 17, 18)})
        stats.append(rec)
        rows.append((index, meta['sample_idx'], p_remap, p_cfg, p_raw))

        # Standalone LIDAR_TOP figure.
        fig, ax = plt.subplots(figsize=(8, 8))
        n = plot_lidar_nuscenes(ax, nusc, meta['sample_idx'], extent)
        ax.set_title(f'LIDAR_TOP + GT boxes ({n}) | {args.split} {index} | {meta["sample_idx"][:10]}')
        fig.savefig(output / f'lidar_top_{index:05d}_{meta["sample_idx"]}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Raw-class channels (no remap, 19 classes): road markings visible separately.
        names = {**NAMES, **RAW_EXTRA}
        cf, caxes = plt.subplots(4, 5, figsize=(20, 16), constrained_layout=True)
        for ch, ax in enumerate(caxes.flat):
            im = ax.imshow(p_raw[ch], origin='lower', extent=extent, interpolation='nearest', cmap='magma', vmin=0, vmax=1)
            ax.set_title(f'{ch}: {names[ch]}' + (' (ignored)' if ch == 19 else ''))
            ax.set_aspect('equal')
        cf.colorbar(im, ax=list(caxes.flat), shrink=.6, label='share (no renormalization)')
        cf.suptitle(f'No remap, raw SAM3 classes (num_classes=19) | {args.split} {index} | {meta["sample_idx"]}')
        cf.savefig(output / f'channels_no_remap_raw_{index:05d}_{meta["sample_idx"]}.png', dpi=130)
        plt.close(cf)
        if args.with_seg:
            save_sample_figure(output, ds, meta, nusc, index, args.split, extent,
                               {k: v for k, v in (('remap', p_remap), ('cfg', p_cfg), ('raw', p_raw)) if k in args.variants})
        print('done', index, meta['sample_idx'], flush=True)

    legend = [Patch(facecolor=UNOBSERVED, label='unobserved')]
    legend += [Patch(facecolor=PALETTE[i], edgecolor='k', label=f'{i}: {n}') for i, n in NAMES.items() if i > 0]
    legend += [Patch(facecolor=PALETTE[i], edgecolor='k', label=f'raw {i}: {n}') for i, n in RAW_EXTRA.items() if i < 19]
    cols = 1 + len(args.variants)
    pages = [rows[i:i + args.rows_per_page] for i in range(0, len(rows), args.rows_per_page)]
    for page_no, page in enumerate(pages):
        fig, axes = plt.subplots(len(page), cols, figsize=(6 * cols, 6 * len(page)), squeeze=False)
        for r, (index, token, p_remap, p_cfg, p_raw) in enumerate(page):
            plot_lidar_nuscenes(axes[r, 0], nusc, token, extent)
            axes[r, 0].set_title(f'{args.split} {index} | {token[:10]}\nLIDAR_TOP + GT boxes')
            probs = dict(remap=p_remap, cfg=p_cfg, raw=p_raw)
            for c, v in enumerate(args.variants, start=1):
                plot_bev(axes[r, c], dominant_rgb(probs[v]), extent, VARIANT_TITLES[v])
        fig.legend(handles=legend, loc='lower center', ncol=10, fontsize=10, frameon=False)
        fig.suptitle('Semantic BEV (detection, LiDAR frame) | dominant class per cell', fontsize=16)
        fig.tight_layout(rect=(0, .05, 1, .98))
        name = 'overview.png' if len(pages) == 1 else f'overview_{page_no:02d}.png'
        fig.savefig(output / name, dpi=110)
        plt.close(fig)
    (output / 'statistics.json').write_text(json.dumps(dict(config=args.config, split=args.split,
        seg_id_remap=remap, samples=stats), indent=2))
    (output / 'README.md').write_text(__doc__.split('Run from')[0] + '''
Files
- overview.png: per sample [LIDAR_TOP + GT | with remap | no remap (config as-is) | no remap (raw classes)].
- lidar_top_*.png: standalone LIDAR_TOP (white background, distance-coloured points, GT boxes).
- channels_no_remap_raw_*.png: all 20 raw-class share channels without remap.
- statistics.json: raw SAM3 pixel counts for ids 16-19, projected pixel mass, observed cells,
  mean road share, cells whose dominant class changes when the remap is removed.
Gray = no class mass (unobserved), not free space. Argmax is only used for plotting.
These maps are fresh projections (TF32 off), not the stored training cache.
''')
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
