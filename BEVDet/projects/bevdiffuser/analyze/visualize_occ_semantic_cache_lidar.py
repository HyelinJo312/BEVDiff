"""Inspect existing ego-frame occupancy semantic caches alongside LIDAR_TOP."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import sys

import numpy as np
from pyquaternion import Quaternion


REPO = Path(__file__).resolve().parents[3]
NAMES = ('unknown / unobserved', 'car', 'road', 'bus', 'truck', 'terrain',
         'vegetation', 'sidewalk', 'bicycle', 'barrier', 'pedestrian',
         'manmade', 'motorcycle', 'construction vehicle', 'trailer', 'traffic cone', 'sky')
COLORS = np.array(((65, 65, 65), (255, 120, 50), (255, 192, 203),
                   (0, 0, 255), (255, 255, 0), (0, 255, 255), (0, 175, 0),
                   (255, 0, 0), (255, 20, 147), (135, 60, 0), (160, 32, 240),
                   (255, 0, 255), (139, 137, 137), (75, 0, 75),
                   (150, 240, 80), (184, 134, 11), (135, 206, 235)), dtype=np.uint8)
BOX_NAMES = {'car': 'car', 'truck': 'truck', 'construction_vehicle': 'construction',
             'bus': 'bus', 'trailer': 'trailer', 'barrier': 'barrier',
             'motorcycle': 'motorcycle', 'bicycle': 'bicycle',
             'pedestrian': 'pedestrian', 'traffic_cone': 'cone'}


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def lidar_to_ego(points, rotation, translation):
    matrix = Quaternion(rotation).rotation_matrix
    offset = np.asarray(translation, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError('Invalid LiDAR-to-ego calibration')
    return np.asarray(points, dtype=np.float64) @ matrix.T + offset


def lidar_box_corners(box):
    from nuscenes.utils.data_classes import Box

    box = np.asarray(box, dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any():
        raise ValueError('Expected a finite center-based nuScenes info box [x,y,z,w,l,h,yaw]')
    orientation = Quaternion(axis=[0, 0, 1], angle=-box[6] - np.pi / 2)
    return Box(box[:3], box[3:6], orientation).corners().T


def gt_boxes_ego(info, bounds):
    from scipy.spatial import ConvexHull

    boxes = np.asarray(info['gt_boxes'])
    names = np.asarray(info['gt_names'])
    valid = np.asarray(info['valid_flag'], dtype=bool)
    if len(boxes) != len(names) or valid.shape != (len(boxes),):
        raise ValueError('GT boxes, names and valid_flag lengths differ')
    records = []
    for index in np.flatnonzero(valid):
        name = str(names[index])
        if name not in BOX_NAMES:
            continue
        corners = lidar_to_ego(lidar_box_corners(boxes[index, :7]),
                               info['lidar2ego_rotation'], info['lidar2ego_translation'])
        if ((corners.max(axis=0) < bounds[:3]) | (corners.min(axis=0) >= bounds[3:])).any():
            continue
        vertices = ConvexHull(corners[:, :2]).vertices
        records.append(dict(annotation_index=int(index), name=name,
                            polygon_xy=corners[vertices, :2].tolist(),
                            corners_ego=corners.tolist()))
    return records


def dominant_labels(probability):
    if probability.ndim != 3 or not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError('Expected finite nonnegative semantic channels')
    labels = probability.argmax(axis=0)
    observed = probability.sum(axis=0) > 0
    labels[~observed] = 0
    return labels, observed


def select_frames(infos, entries, count, generator):
    scenes = {}
    for info in sorted(infos, key=lambda entry: entry['timestamp']):
        if info['token'] in entries:
            scenes.setdefault(info['scene_token'], []).append(info)
    if len(scenes) < count:
        raise ValueError(f'Need {count} cached scenes, found {len(scenes)}')
    keys = sorted(scenes)
    chosen = generator.choice(len(keys), size=count, replace=False)
    return [scenes[keys[index]][len(scenes[keys[index]]) // 2] for index in chosen]


def sensor_path(value):
    path = Path(value)
    candidates = [path] if path.is_absolute() else [REPO / path, REPO / 'BEVFormer' / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot resolve sensor path {value}')


def draw_sample(axes, record, points, labels, extent, bounds, plt):
    import matplotlib.patheffects as effects
    from matplotlib.patches import Polygon

    axes[0].set_facecolor('white')
    cloud = axes[0].scatter(points[:, 0], points[:, 1], c=points[:, 2],
                            cmap='turbo', vmin=bounds[2], vmax=bounds[5],
                            s=.35, linewidths=0, rasterized=True)
    plt.colorbar(cloud, ax=axes[0], fraction=.035, pad=.025, label='ego z [m]')
    axes[1].imshow(COLORS[labels], origin='lower', extent=extent, interpolation='nearest')
    axes[2].imshow(COLORS[labels], origin='lower', extent=extent, interpolation='nearest')
    axes[2].scatter(points[:, 0], points[:, 1], s=.25, c='white', alpha=.45,
                    linewidths=0, rasterized=True)
    titles = (f'LIDAR_TOP + GT boxes ({record["gt_boxes_in_range"]} boxes)',
              f'Semantic argmax + GT ({record["observed_percent"]:.1f}% observed)',
              'Semantic cache + LiDAR + GT boxes')
    for column, (axis, title) in enumerate(zip(axes, titles)):
        axis.set(xlim=extent[:2], ylim=extent[2:], aspect='equal',
                 xlabel='ego x [m] (forward)', ylabel='ego y [m] (left)', title=title)
        axis.plot(0, 0, '+', color='black' if column == 0 else 'white',
                  markersize=9, markeredgewidth=1.6)
        axis.set_xticks(np.arange(-40, 41, 20))
        axis.set_yticks(np.arange(-40, 41, 20))
        for box in record['gt_boxes']:
            polygon = np.asarray(box['polygon_xy'])
            patch = Polygon(polygon, closed=True, fill=False, edgecolor='black',
                            linewidth=.9, zorder=5)
            patch.set_path_effects([effects.Stroke(linewidth=2.1, foreground='white'),
                                    effects.Normal()])
            axis.add_patch(patch)
            if column == 0:
                anchor = polygon.mean(axis=0)
                if bounds[0] <= anchor[0] < bounds[3] and bounds[1] <= anchor[1] < bounds[4]:
                    axis.text(anchor[0], anchor[1], BOX_NAMES[box['name']], fontsize=5.5,
                              color='black', clip_on=True, zorder=6,
                              bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.2))
    axes[0].text(.02, .98, f'{record["split"]} | scene {record["scene_token"][:8]}\n'
                 f'{record["token"][:12]}', transform=axes[0].transAxes, va='top',
                 fontsize=8, color='black', bbox=dict(facecolor='white', alpha=.8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=REPO / 'BEVFormer/data/semantic_bev_occ_v2')
    parser.add_argument('--annotation-root', type=Path, default=REPO / 'BEVFormer/data/occ3d-nus')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-split', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--overwrite', action='store_true',
                        help='Redraw existing selected samples in the output directory')
    args = parser.parse_args()
    if not 1 <= args.per_split <= 6:
        parser.error('--per-split must be between 1 and 6 for this bounded inspection')
    output = args.output.resolve()
    previous = None
    if output.exists() and not args.overwrite:
        parser.error(f'Refusing to overwrite an existing output directory: {output}')
    if output.exists():
        previous = json.loads((output / 'provenance.json').read_text())
        expected_manifest = args.cache_root.resolve() / 'manifest.json'
        if Path(previous['manifest_path']).resolve() != expected_manifest:
            parser.error('Overwrite must use the same cache as the existing visualization')
    manifest_path = args.cache_root.resolve() / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    projection = manifest['contract']['projection']
    if projection.get('output_frame') != 'ego' or projection['num_classes'] != 16:
        parser.error('Expected an ego-frame, 17-channel semantic cache')
    bounds = np.asarray(projection['pc_range'], dtype=np.float64)
    extent = (bounds[0], bounds[3], bounds[1], bounds[4])
    expected_shape = (17, projection['bev_h'], projection['bev_w'])
    generator = np.random.default_rng(args.seed)
    selected, annotations = [], {}
    for split in ('train', 'val'):
        annotation = args.annotation_root / f'occ_infos_temporal_{split}.pkl'
        with annotation.open('rb') as stream:
            payload = pickle.load(stream)
        if previous is None:
            chosen = select_frames(payload['infos'], manifest['entries'], args.per_split, generator)
        else:
            by_token = {info['token']: info for info in payload['infos']}
            chosen = [by_token[row['token']] for row in previous['samples'] if row['split'] == split]
        selected.extend((split, info) for info in chosen)
        annotations[split] = dict(path=str(annotation.resolve()), sha256=sha256(annotation),
                                  annotation_samples=len(payload['infos']))
        del payload
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output.mkdir(parents=True, exist_ok=args.overwrite)
    records, displays = [], []
    for split, info in selected:
        token = info['token']
        cache_path = args.cache_root.resolve() / f'{token}.npy'
        probability = np.load(cache_path, allow_pickle=False)
        if probability.shape != expected_shape or probability.dtype != np.float32:
            raise ValueError(f'Unexpected cache shape/dtype: {cache_path}')
        checksum = hashlib.sha256(probability.tobytes()).hexdigest()
        if checksum != manifest['entries'][token]['sha256']:
            raise ValueError(f'Cache checksum mismatch: {cache_path}')
        labels, observed = dominant_labels(probability)
        lidar_path = sensor_path(info['lidar_path'])
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        ego = lidar_to_ego(raw, info['lidar2ego_rotation'], info['lidar2ego_translation'])
        valid = np.isfinite(ego).all(axis=1) & ((ego >= bounds[:3]) & (ego < bounds[3:])).all(axis=1)
        points = ego[valid]
        boxes = gt_boxes_ego(info, bounds)
        record = dict(split=split, token=token, scene_token=info['scene_token'],
                      timestamp=int(info['timestamp']), cache_path=str(cache_path),
                      cache_sha256=checksum, checksum_passed=True, lidar_path=str(lidar_path),
                      lidar_file_sha256=sha256(lidar_path), lidar_points_total=len(raw),
                      lidar_points_in_xyz_range=int(valid.sum()), observed_cells=int(observed.sum()),
                      observed_percent=float(observed.mean() * 100),
                      lidar2ego_rotation=info['lidar2ego_rotation'],
                      lidar2ego_translation=info['lidar2ego_translation'],
                      gt_boxes_total=len(info['gt_boxes']),
                      gt_boxes_in_range=len(boxes), gt_boxes=boxes,
                      dominant_class_cells={NAMES[index]: int((labels == index).sum())
                                            for index in range(len(NAMES))})
        records.append(record)
        displays.append((points, labels))
        print(f'{split} {token}: checksum OK; {len(points):,} LiDAR points; '
              f'{record["observed_percent"]:.1f}% observed cells', flush=True)
    present = sorted(set(np.concatenate([labels.ravel() for _, labels in displays]).tolist()))
    legend = [Patch(facecolor=COLORS[index] / 255., label=f'{index}: {NAMES[index]}') for index in present]
    legend.append(Line2D([0], [0], color='black', linewidth=1, label='GT 3D box (XY projection)'))
    for record, (points, labels) in zip(records, displays):
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        draw_sample(axes, record, points, labels, extent, bounds, plt)
        fig.suptitle(f'{record["split"]} | {record["token"]} | ego-frame semantic BEV', fontsize=12)
        fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=8, frameon=False)
        fig.subplots_adjust(left=.05, right=.99, top=.86, bottom=.23, wspace=.30)
        fig.savefig(output / f'{record["split"]}_{record["token"]}.png', dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(len(records), 3, figsize=(16, len(records) * 4.7), squeeze=False)
    for row, (record, (points, labels)) in enumerate(zip(records, displays)):
        draw_sample(axes[row], record, points, labels, extent, bounds, plt)
    fig.suptitle('Occupancy semantic BEV v2 | train + validation | all panels in ego frame', y=.994)
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=9, frameon=False)
    fig.subplots_adjust(left=.05, right=.99, top=.98, bottom=.07, wspace=.30, hspace=.32)
    fig.savefig(output / 'overview.png', dpi=140)
    plt.close(fig)
    mode = 'w' if args.overwrite else 'x'
    with (output / 'selected_samples.csv').open(mode, newline='') as stream:
        fields = ('split', 'token', 'scene_token', 'lidar_points_total',
                  'lidar_points_in_xyz_range', 'observed_cells', 'observed_percent',
                  'gt_boxes_total', 'gt_boxes_in_range', 'checksum_passed')
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    provenance = dict(command=shlex.join([sys.executable, *sys.argv]), cwd=str(Path.cwd()),
                      seed=previous['seed'] if previous is not None else args.seed,
                      script_sha256=sha256(__file__), manifest_path=str(manifest_path),
                      manifest_sha256=sha256(manifest_path), stored_contract=manifest['contract'],
                      manifest_entry_count=len(manifest['entries']), annotations=annotations,
                      selection='seeded uniform distinct scenes per split; middle cached frame per scene',
                      selection_reused=previous is not None,
                      gt_box_policy='valid_flag=True, standard 10 detection classes, ego XYZ AABB intersects range; project all eight 3D corners to XY convex hull',
                      gt_yaw_convention='legacy nuScenes info: SDK heading = -stored_yaw - pi/2; stored size = w,l,h; stored origin = center',
                      validation_scope='selected-array checksums/shapes/values only; no geometry-hash or live projection validation',
                      samples=records)
    with (output / 'provenance.json').open(mode) as stream:
        json.dump(provenance, stream, indent=2)
    with (output / 'README.md').open(mode) as stream:
        stream.write('# Occupancy semantic BEV v2 + LIDAR_TOP\n\n'
                     '- overview.png: all selected samples; individual PNGs show one sample each.\n'
                     '- Columns: white-background LiDAR colored by ego height + GT; semantic argmax + GT; semantic + white LiDAR overlay + GT.\n'
                     '- All panels use ego coordinates: horizontal x forward, vertical y left; the cross is the ego origin.\n'
                     '- LiDAR is transformed by the annotation LiDAR-to-ego rotation/translation BEFORE XYZ clipping.\n'
                     '- Black GT outlines with a white halo are the XY convex hull of all eight transformed 3D box corners; labels appear on the LiDAR panel.\n'
                     '- Boxes use valid_flag=True (LiDAR or radar support), the standard 10 detection classes, and ego XYZ range intersection. This is a display policy, not occupancy voxel GT.\n'
                     '- Annotation boxes store center XYZ, width/length/height and legacy SECOND yaw. SDK heading = -stored yaw - pi/2. Full 3D corners are transformed to ego, including calibration roll/pitch.\n'
                     f'- XYZ range: {bounds.tolist()}; semantic grid: {expected_shape[1:]}.\n'
                     '- Gray semantic cells have zero total semantic mass: unknown/unobserved, NOT confirmed free space.\n'
                     '- Argmax and fixed class colors are for visualization only; no thresholding, smoothing, resizing, or cache edits.\n'
                     '- Semantic colors are predictions from the cached SAM3/depth prior, not LiDAR semantic ground truth.\n'
                     '- No raw SAM3/Metric3D reads, GPU execution, cache regeneration, or training.\n'
                     '- Selected arrays passed manifest SHA256, shape, dtype, finite/nonnegative checks.\n'
                     '- This does NOT establish full-cache completeness, current-code equivalence, geometry-hash validation, or downstream accuracy.\n'
                     '- selected_samples.csv and provenance.json record the deterministic selection, counts, transforms, and source hashes.\n')
    print(f'Saved visualization to {output}', flush=True)


if __name__ == '__main__':
    main()
