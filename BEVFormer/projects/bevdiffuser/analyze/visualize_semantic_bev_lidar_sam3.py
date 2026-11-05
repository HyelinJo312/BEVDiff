"""Cached semantic BEV next to a LiDAR top view with GT boxes and SAM3-overlaid camera images.

Read-only. Run from BEVFormer/ in jhl-bevdiff:
    python projects/bevdiffuser/analyze/visualize_semantic_bev_lidar_sam3.py --indices 0 1500 3000
Writes one PNG per val sample to results/visualize/semantic_bev_lidar_sam3/.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from PIL import Image

BEVFORMER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BEVFORMER))
sys.path.insert(0, str(BEVFORMER / 'projects/bevdiffuser'))  # debug_seg_bev imports model_utils
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.bevdiffuser.analyze.debug_seg_bev import CLASS_COLORS, SEG_CLASS_NAMES

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
EXTENT = [PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]]
# Same raw SAM3 -> model taxonomy remap the cache was built with (manifest contract):
# crosswalk(16)/lane(17)/road arrow(18) -> highway/road(2), sky(19) -> 16.
SEG_ID_REMAP = {16: 2, 17: 2, 18: 2, 19: 16}
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
BOX_COLORS = {'car': '#1f77b4', 'truck': '#ff7f0e', 'construction_vehicle': '#8c564b', 'bus': '#d62728',
              'trailer': '#9467bd', 'barrier': '#7f7f7f', 'motorcycle': '#e377c2', 'bicycle': '#bcbd22',
              'pedestrian': '#2ca02c', 'traffic_cone': '#17becf'}
COLORS = np.array(CLASS_COLORS)  # (17, 3) RGB in [0, 1]


def semantic_bev_rgb(prob):
    """Solid colour of the per-cell argmax class; unobserved (zero-mass) cells stay white."""
    mass = prob.sum(0)
    rgb = COLORS[prob.argmax(0)]
    rgb[mass <= 1e-6] = 1.0
    return rgb, mass


def remap(raw):
    out = raw.copy()
    for source, target in SEG_ID_REMAP.items():
        out[raw == source] = target
    return out


def lidar_panel(ax, info, data_root):
    points = np.fromfile(data_root / info['lidar_path'], dtype=np.float32).reshape(-1, 5)
    keep = ((points[:, 0] >= EXTENT[0]) & (points[:, 0] < EXTENT[1]) &
            (points[:, 1] >= EXTENT[2]) & (points[:, 1] < EXTENT[3]))
    points = points[keep]
    ax.scatter(points[:, 0], points[:, 1], s=0.4, c='#404040', linewidths=0, rasterized=True)
    boxes = info['gt_boxes'][info['valid_flag']]
    names = info['gt_names'][info['valid_flag']]
    if len(boxes):
        corners = LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1], origin=(0.5, 0.5, 0.5)).corners.numpy()
        for corner, name in zip(corners, names):
            footprint = corner[[0, 3, 7, 4], :2]  # bottom face in LiDAR XY
            color = BOX_COLORS.get(name, 'black')
            ax.add_patch(Polygon(footprint, closed=True, fill=False, edgecolor=color, linewidth=1.0))
            front = footprint[2:4].mean(0)  # mark heading with a line from the centre to the front edge
            centre = footprint.mean(0)
            ax.plot([centre[0], front[0]], [centre[1], front[1]], color=color, linewidth=0.8)
    ax.plot(0, 0, marker='^', color='red', markersize=6)
    ax.set(xlim=EXTENT[:2], ylim=EXTENT[2:], aspect='equal', facecolor='white',
           title=f'LiDAR top view + GT boxes ({int(info["valid_flag"].sum())})',
           xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)')
    ax.legend(handles=[Patch(edgecolor=c, fill=False, label=n) for n, c in BOX_COLORS.items() if n in set(names)],
              loc='upper right', fontsize=6, framealpha=0.8)


def camera_panel(ax, info, cam, data_root, sam3_root, alpha=0.5):
    path = info['cams'][cam]['data_path']
    image = np.asarray(Image.open(data_root / path).convert('RGB'), dtype=np.float32) / 255.0
    rel = path.split('samples/')[-1]
    raw = np.fromfile(sam3_root / 'samples' / (rel[:-len('.jpg')] + '_mask.bin'), dtype=np.int8)
    seg = remap(raw.reshape(image.shape[:2]).astype(np.int16))
    labelled = (seg > 0) & (seg < len(COLORS))
    overlay = image.copy()
    overlay[labelled] = (1 - alpha) * image[labelled] + alpha * COLORS[seg[labelled]]
    ax.imshow(overlay)
    ax.set_title(cam, fontsize=8)
    ax.axis('off')
    return set(np.unique(seg[labelled]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info', default=str(BEVFORMER / 'data/nuscenes/nuscenes_infos_temporal_val.pkl'))
    parser.add_argument('--cache-root', default=str(BEVFORMER / 'data/semantic_bev'))
    parser.add_argument('--sam3-root', default=str(BEVFORMER / 'data/nuscenes_sam3'))
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 1500, 3000, 4500, 6000])
    parser.add_argument('--output-dir', default=str(BEVFORMER.parent / 'results/visualize/semantic_bev_lidar_sam3'))
    args = parser.parse_args()

    infos = pickle.load(open(args.info, 'rb'))['infos']
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_root, sam3_root, cache_root = BEVFORMER, Path(args.sam3_root), Path(args.cache_root)

    for index in args.indices:
        info = infos[index]
        prob = np.load(cache_root / f'{info["token"]}.npy', allow_pickle=False)
        rgb, mass = semantic_bev_rgb(prob)

        fig = plt.figure(figsize=(18, 12.5))
        grid = fig.add_gridspec(3, 3, height_ratios=[2.2, 1, 1], hspace=0.15, wspace=0.08, top=0.93)
        ax_bev = fig.add_subplot(grid[0, 0])
        ax_bev.imshow(rgb, origin='lower', extent=EXTENT, interpolation='nearest')
        ax_bev.plot(0, 0, marker='^', color='red', markersize=6)
        ax_bev.set(title=f'Cached semantic BEV (argmax, {(mass > 1e-6).mean():.0%} cells observed)',
                   xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', aspect='equal')
        ax_mass = fig.add_subplot(grid[0, 1])
        lidar_panel(ax_mass, info, data_root)
        ax_obs = fig.add_subplot(grid[0, 2])
        im = ax_obs.imshow(mass, origin='lower', extent=EXTENT, cmap='viridis', vmin=0, vmax=1,
                           interpolation='nearest')
        ax_obs.set(title='Per-cell semantic mass (sum over classes)', xlabel='LiDAR X (m)', aspect='equal')
        fig.colorbar(im, ax=ax_obs, fraction=0.046, pad=0.02)

        seen = set(np.nonzero(prob.sum((1, 2)) > 0)[0].tolist())
        for k, cam in enumerate(CAMS):
            ax = fig.add_subplot(grid[1 + k // 3, k % 3])
            seen |= camera_panel(ax, info, cam, data_root, sam3_root)
        handles = [Patch(color=COLORS[c], label=f'{c} {SEG_CLASS_NAMES[c]}') for c in sorted(seen) if c > 0]
        fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))
        fig.suptitle(f'val[{index}] token={info["token"]}  (cache: {cache_root.name}, remap {SEG_ID_REMAP})',
                     y=0.955, fontsize=10)
        path = output / f'val{index:04d}_{info["token"]}.png'
        fig.savefig(path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print('saved', path)


if __name__ == '__main__':
    main()
