"""Visualize the exact pre-embedding v5 semantic probabilities, without weights.

Run with the jhl-bevdiff environment. Defaults to six validation scenes and
workspace/visualize/metric3d_v5. NPZ arrays retain [C, Y, X] orientation and the
original semantic mass (no display-only renormalization).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter, NullLocator
from mmcv import Config

BEVFORMER = Path(__file__).resolve().parents[2]
WORKSPACE = BEVFORMER.parent
sys.path.insert(0, str(BEVFORMER))

import projects.mmdet3d_plugin  # Register datasets before importing data_utils.
from mmdet3d.datasets import build_dataset
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner
from debug_seg_bev import CLASS_COLORS, SEG_CLASS_NAMES

COLORS = np.asarray(CLASS_COLORS, dtype=np.float32)
UNKNOWN = np.array([.15, .15, .15], dtype=np.float32)
CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def draw_bev(ax, values, extent, title, **kwargs):
    rendered = ax.imshow(values, origin='lower', extent=extent,
                         interpolation='nearest', **kwargs)
    ax.plot(0, 0, '+', color='white', markersize=7, markeredgewidth=1)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)')
    ax.set_aspect('equal')
    return rendered


def class_rgb(probabilities):
    valid = probabilities.sum(axis=0) > 0
    rgb = COLORS[probabilities.argmax(axis=0)]
    rgb[~valid] = UNKNOWN
    return rgb, valid


def draw_gt_boxes(ax, boxes, extent):
    """Draw rotated LiDAR-frame bottom faces and their positive-X heading."""
    if boxes is None or len(boxes) == 0:
        return 0
    footprints = boxes.corners[:, [0, 3, 7, 4], :2].cpu().numpy()
    count = 0
    for corners in footprints:
        if (corners[:, 0].max() < extent[0] or corners[:, 0].min() > extent[1]
                or corners[:, 1].max() < extent[2] or corners[:, 1].min() > extent[3]):
            continue
        ax.add_patch(mpatches.Polygon(corners, closed=True, fill=False,
                                     edgecolor='#d62728', linewidth=1.2, zorder=4))
        center = corners.mean(axis=0)
        front = corners[2:].mean(axis=0)
        ax.plot([center[0], front[0]], [center[1], front[1]],
                color='#d62728', linewidth=1.0, zorder=4)
        count += 1
    return count


def save_diagnostics(probabilities, support, extent, lidar_path, prefix, title,
                     gt_boxes=None, lidar_stride=1, lidar_point_size=.35):
    rgb, valid = class_rgb(probabilities)
    total = probabilities.sum(axis=0)
    blended = np.einsum('chw,cd->hwd', probabilities, COLORS[:len(probabilities)])
    blended += np.clip(1 - total, 0, 1)[..., None] * UNKNOWN
    fig, axes = plt.subplots(2, 3, figsize=(17, 12), constrained_layout=True)
    ax = axes[0, 0]
    if Path(lidar_path).is_file():
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        inside = ((points[:, 0] >= extent[0]) & (points[:, 0] < extent[1])
                  & (points[:, 1] >= extent[2]) & (points[:, 1] < extent[3]))
        points = points[inside][::lidar_stride]
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.grid(True, color='#dddddd', linewidth=.5)
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis',
                   s=lidar_point_size, alpha=1.0, linewidths=0,
                   vmin=-3, vmax=3, rasterized=True)
        box_count = draw_gt_boxes(ax, gt_boxes, extent)
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='LiDAR X (m)',
               ylabel='LiDAR Y (m)', title=f'LiDAR top + GT objects ({box_count})')
        ax.set_aspect('equal')
        ax.plot(0, 0, '+', color='black', zorder=5)
        ax.legend(handles=[mpatches.Patch(facecolor='none', edgecolor='#d62728',
                                         label='GT bbox + heading')],
                  loc='upper right', fontsize=8, facecolor='white', framealpha=.9)
    else:
        ax.text(.5, .5, 'LiDAR file unavailable', ha='center', transform=ax.transAxes)
        ax.axis('off')
    draw_bev(axes[0, 1], rgb, extent, f'Dominant class | observed {valid.mean():.1%}')
    draw_bev(axes[0, 2], np.clip(blended, 0, 1), extent, 'Probability-weighted class colors')
    for ax, values, label, cmap, limits in (
        (axes[1, 0], probabilities.max(axis=0), 'Maximum semantic mass', 'magma', (0, 1)),
        (axes[1, 1], np.log1p(support), 'log(1 + support pixel mass)', 'viridis', (0, None)),
        (axes[1, 2], total, 'Sum of semantic mass', 'cividis', (0, 1)),
    ):
        rendered = draw_bev(ax, values, extent, label, cmap=cmap, vmin=limits[0], vmax=limits[1])
        fig.colorbar(rendered, ax=ax, shrink=.8)
    present = np.unique(probabilities.argmax(axis=0)[valid])
    handles = [mpatches.Patch(color=COLORS[c], label=f'{c}: {SEG_CLASS_NAMES[c]}') for c in present]
    handles.append(mpatches.Patch(color=UNKNOWN, label='Unobserved'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 0), ncol=6, fontsize=8)
    fig.suptitle(title + '\nRaw v5 projection; support is not depth confidence', fontsize=12)
    fig.savefig(str(prefix) + '_bev.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    channels = list(range(1, probabilities.shape[0]))
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), constrained_layout=True)
    for ax, channel in zip(axes.flat, channels):
        rendered = draw_bev(ax, probabilities[channel], extent,
                            f'{channel}: {SEG_CLASS_NAMES[channel]}', cmap='magma', vmin=0, vmax=1)
    for ax in list(axes.flat)[len(channels):]:
        ax.axis('off')
    fig.colorbar(rendered, ax=axes, shrink=.65, label='Raw semantic mass (same scale for all classes)')
    fig.suptitle(title + '\nClass channels before prob_to_emb', fontsize=12)
    fig.savefig(str(prefix) + '_classes.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_inputs(item, meta, seg, depth, prefix, depth_range=(2.5, 170.0), depth_scale='log'):
    # Use the pipeline image, avoiding a stretch of its padding into raw RGB.
    images = item['img'][0].data.numpy().transpose(0, 2, 3, 1)
    norm = meta['img_norm_cfg']
    images = images * np.asarray(norm['std']) + np.asarray(norm['mean'])
    if not norm.get('to_rgb', False):
        images = images[..., ::-1]
    images = np.clip(images / 255., 0, 1)
    view_by_name = {Path(filename).parent.name: view for view, filename in enumerate(meta['filename'])}
    depth_norm = (LogNorm if depth_scale == 'log' else Normalize)(*depth_range)
    depth_cmap = plt.get_cmap('turbo_r').copy()
    depth_cmap.set_bad('#ededed')
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), constrained_layout=True)
    for position, name in enumerate(CAMERAS):
        view = view_by_name[name]
        row, col = 2 * (position // 3), position % 3
        mask = seg[view]
        valid = (mask > 0) & (mask < len(COLORS))
        overlay = images[view].copy()
        overlay[valid] = .55 * overlay[valid] + .45 * COLORS[mask[valid]]
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(name + ' | SAM3 overlay', fontsize=10)
        rendered = axes[row + 1, col].imshow(np.ma.masked_less_equal(depth[view], 0),
                                              cmap=depth_cmap, norm=depth_norm,
                                              interpolation='nearest')
        axes[row + 1, col].set_title('Metric3D depth | near: red, far: blue', fontsize=10)
    for ax in axes.flat:
        ax.axis('off')
    ticks = sorted(set([depth_range[0], depth_range[1]]
                       + [value for value in (5, 10, 20, 40, 80) if depth_range[0] < value < depth_range[1]]))
    colorbar = fig.colorbar(rendered, ax=axes, shrink=.65, ticks=ticks,
                           format=FormatStrFormatter('%g'),
                           label=f'Depth (m, {depth_scale} scale); gray = invalid / padding')
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    fig.savefig(str(prefix) + '_inputs.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bev_config', default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--output_dir', default=str(WORKSPACE / 'visualize/metric3d_v5'))
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--indices', type=int, nargs='+', help='Explicit validation dataset indices')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pixel_stride', type=int, help='Override semantic splatting stride without editing the config')
    parser.add_argument('--lidar_stride', type=int, default=1, help='Display one point per N in-range LiDAR points')
    parser.add_argument('--lidar_point_size', type=float, default=.35, help='Marker area in points squared')
    parser.add_argument('--depth_scale', choices=('log', 'linear'), default='log')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num_samples must be positive')
    if args.pixel_stride is not None and args.pixel_stride < 1:
        parser.error('--pixel_stride must be positive')
    if args.lidar_stride < 1 or not np.isfinite(args.lidar_point_size) or args.lidar_point_size <= 0:
        parser.error('LiDAR stride and point size must be positive')
    cfg = Config.fromfile(args.bev_config)
    if args.pixel_stride is not None:
        cfg.unet.parameters.seg_bev_aligner.pixel_stride = args.pixel_stride
    settings = cfg.data.test.copy()
    # These diagnostics also render raw depth and support, which are not cached.
    settings.update(test_mode=True, load_annos=True, use_semantic_bev_cache=False)
    dataset = build_dataset(settings, default_args=dict(
        pc_range=cfg.point_cloud_range, use_3d_bbox=cfg.use_3d_bbox,
        num_classes=cfg.num_classes, num_bboxes=cfg.num_bboxes))
    aligner = SegBEVAligner(**cfg.unet.parameters.seg_bev_aligner).to(args.device).eval()
    indices, seen = [], set()
    if args.indices is not None:
        indices = args.indices
    else:
        for index, info in enumerate(dataset.data_infos):
            if info['scene_token'] not in seen:
                indices.append(index)
                seen.add(info['scene_token'])
            if len(indices) >= args.num_samples:
                break
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = tuple(cfg.point_cloud_range[i] for i in (0, 3, 1, 4))
    summary, previews = [], []
    for index in indices:
        item = dataset[index]
        meta = dataset._current_metas(item['img_metas'])
        seg = item['seg_maps'].data.unsqueeze(0).to(args.device)
        depth = item['depth_maps'].data.unsqueeze(0).to(args.device)
        probabilities, support = aligner.project_semantics(seg, [meta], depth)
        probabilities = probabilities[0].cpu().numpy()
        support = support[0, 0].cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.isfinite(support).all():
            raise ValueError(f'Non-finite projection in sample {index}')
        token = meta['sample_idx']
        prefix = output_dir / f'sample_{index:05d}_{token[:8]}'
        title = (f'Metric3D v5 | stride {aligner.pixel_stride} | '
                 f'index {index} | sample {token[:12]}')
        lidar_path = dataset.get_data_info(index)['pts_filename']
        annotations = dataset.get_ann_info(index)
        gt_boxes = annotations['gt_bboxes_3d'][annotations['gt_labels_3d'] >= 0]
        save_diagnostics(probabilities, support, extent, lidar_path, prefix, title, gt_boxes=gt_boxes,
                         lidar_stride=args.lidar_stride, lidar_point_size=args.lidar_point_size)
        save_inputs(item, meta, seg[0].cpu().numpy(), depth[0].cpu().numpy(), prefix,
                    depth_range=aligner.depth_range, depth_scale=args.depth_scale)
        rgb, valid = class_rgb(probabilities)
        np.savez_compressed(str(prefix) + '.npz', probabilities=probabilities, support=support,
                            valid=valid, pc_range=np.asarray(cfg.point_cloud_range),
                            class_names=np.asarray([SEG_CLASS_NAMES[i] for i in range(len(probabilities))]),
                            sample_token=token, scene_token=meta['scene_token'],
                            pixel_stride=aligner.pixel_stride)
        summary.append(dict(index=index, sample_token=token, scene_token=meta['scene_token'],
                            observed_fraction=float(valid.mean()), prefix=prefix.name,
                            probability_shape=list(probabilities.shape)))
        previews.append((rgb, title, valid.mean()))
        print(f'Saved {prefix.name}: observed={valid.mean():.1%}', flush=True)
    rows = (len(previews) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, rows * 6), squeeze=False, constrained_layout=True)
    for ax, (rgb, title, coverage) in zip(axes.flat, previews):
        draw_bev(ax, rgb, extent, title + f'\nObserved {coverage:.1%}')
    for ax in list(axes.flat)[len(previews):]:
        ax.axis('off')
    handles = [mpatches.Patch(color=COLORS[c], label=f'{c}: {SEG_CLASS_NAMES[c]}')
               for c in range(1, aligner.num_classes + 1) if c != aligner.sky_id]
    handles.append(mpatches.Patch(color=UNKNOWN, label='Unobserved'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 0), ncol=6, fontsize=8)
    fig.savefig(output_dir / 'overview.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    report = dict(config=str(Path(args.bev_config).resolve()),
                  aligner=dict(cfg.unet.parameters.seg_bev_aligner),
                  visualization=dict(lidar_stride=args.lidar_stride, lidar_point_size=args.lidar_point_size,
                                     lidar_alpha=1.0, depth_scale=args.depth_scale, depth_cmap='turbo_r'),
                  note='Raw pre-embedding semantic mass. No renormalization or learned weights. '
                       'LiDAR X horizontal, Y vertical, origin lower. Support is not depth confidence.',
                  samples=summary)
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Results: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
