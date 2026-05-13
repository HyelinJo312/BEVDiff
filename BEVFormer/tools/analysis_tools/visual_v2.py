# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Extended for baseline-vs-ours comparison: separate camera files per model + GT,
#  6-surround grid packed with zero whitespace.
# ---------------------------------------------

import argparse
import os
import pickle

import mmcv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

import matplotlib.patches as mpatches

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name


# Camera order for the 2x3 surround grid (top row: front, bottom row: back)
SURROUND_CAMS = [
    'CAM_FRONT_LEFT',  'CAM_FRONT',  'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',   'CAM_BACK',   'CAM_BACK_RIGHT',
]

# 10-class nuScenes detection set (matches BEVFormer/diff_bevformer configs).
NUSC_DET_CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)
# Matches `point_cloud_range` in projects/configs/diff_bevformer/layout_tiny*.py
DEFAULT_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


# --------------------------------------------------------------------------- #
# Training-time GT replication (from val info pkl)
# --------------------------------------------------------------------------- #
def _load_gt_infos(pkl_path):
    """Load nuScenes info pkl -> {sample_token: info_dict}."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    return {item['token']: item for item in infos}


def _gt_keep_mask(info, gt_classes, pc_range):
    """
    Replicate training-time GT filtering as bool mask over `info['gt_boxes']`:
        valid_flag  AND  gt_name in gt_classes  AND  center within pc_range.

    pc_range matches mmdet3d's ObjectRangeFilter (strict `<`/`>` on box center
    in LIDAR sensor frame).
    """
    valid = np.asarray(info['valid_flag'], dtype=bool)
    names = np.asarray(info['gt_names'])
    in_cls = np.array([n in gt_classes for n in names], dtype=bool)
    xyz = np.asarray(info['gt_boxes'])[:, :3]
    in_range = (
        (xyz[:, 0] > pc_range[0]) & (xyz[:, 0] < pc_range[3]) &
        (xyz[:, 1] > pc_range[1]) & (xyz[:, 1] < pc_range[4]) &
        (xyz[:, 2] > pc_range[2]) & (xyz[:, 2] < pc_range[5])
    )
    return valid & in_cls & in_range


def _filtered_ann_tokens(sample, info, gt_classes, pc_range):
    """
    Map filter mask back to `sample['anns']`. The nuScenes info-pkl converter
    iterates `sample['anns']` in order to build `gt_boxes`/`gt_names`/...,
    so element i of those arrays corresponds to `sample['anns'][i]`.
    """
    anns = list(sample['anns'])
    if info is None:
        return anns  # no pkl -> no filtering (legacy behavior)
    mask = _gt_keep_mask(info, gt_classes, pc_range)
    if len(anns) != len(mask):
        # Defensive fallback: if the converter ever drops anns, do not filter.
        return anns
    return [a for a, keep in zip(anns, mask) if keep]


# --------------------------------------------------------------------------- #
# Coordinate helpers (predicted-box version of nusc.get_sample_data)
# --------------------------------------------------------------------------- #
def _get_predicted_data(nusc: NuScenes,
                        sample_data_token: str,
                        pred_anns,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        use_flat_vehicle_coordinates: bool = False):
    """Project predicted Box list (in global frame) into the given sensor frame."""
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    box_list = []
    for box in pred_anns:
        if use_flat_vehicle_coordinates:
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2),
                                  vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # global -> ego
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            # ego -> sensor
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)
    return data_path, box_list, cam_intrinsic


# --------------------------------------------------------------------------- #
# Color helper (per-class color for camera-view box rendering)
# --------------------------------------------------------------------------- #
def _get_color(nusc: NuScenes, category_name: str):
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    if category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    if category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']
    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


# --------------------------------------------------------------------------- #
# BEV (LiDAR top-down) renderer using the official nuScenes utility
# --------------------------------------------------------------------------- #
def _build_eval_boxes(nusc, sample_token, gt=False, pred_records=None):
    """Build an EvalBoxes container of DetectionBox for either GT or predictions."""
    box_list = []
    if gt:
        anns = nusc.get('sample', sample_token)['anns']
        for ann in anns:
            content = nusc.get('sample_annotation', ann)
            try:
                box_list.append(DetectionBox(
                    sample_token=content['sample_token'],
                    translation=tuple(content['translation']),
                    size=tuple(content['size']),
                    rotation=tuple(content['rotation']),
                    velocity=nusc.box_velocity(content['token'])[:2],
                    ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                    else tuple(content['ego_translation']),
                    num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                    detection_name=category_to_detection_name(content['category_name']),
                    detection_score=-1.0 if 'detection_score' not in content
                    else float(content['detection_score']),
                    attribute_name=''))
            except Exception:
                pass
    else:
        for content in pred_records:
            box_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=tuple(content['velocity']),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=content['detection_name'],
                detection_score=-1.0 if 'detection_score' not in content
                else float(content['detection_score']),
                attribute_name=content['attribute_name']))
    eb = EvalBoxes()
    eb.add_boxes(sample_token, box_list)
    return eb


def render_bev(nusc, sample_token, pred_results, out_path,
               score_thr=0.2, axes_limit=50, dpi=200, figsize=(8, 8),
               gt_ann_tokens=None):
    """BEV LiDAR top-down: GT (green) vs predicted (blue) boxes.

    Axes ticks/labels are hidden; a legend is shown in the upper-right corner.
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # LiDAR points: keep in sensor frame — matches visualize_sample (nsweeps=1).
    # from_file_multisweep with single sweep applies identity net-transform,
    # so loading raw sensor-frame points is equivalent.
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
    pts = pc.points  # (4, N) in LIDAR_TOP sensor frame
    dists = np.sqrt(pts[0] ** 2 + pts[1] ** 2)
    colors = np.minimum(1.0, dists / axes_limit)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.scatter(pts[0], pts[1], c=colors, s=0.2, cmap='viridis',
               linewidths=0, rasterized=True)
    ax.plot(0, 0, 'x', color='black')
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_aspect('equal')
    ax.axis('off')

    def _global_to_sensor(translation, size, rotation):
        box = Box(translation, size, Quaternion(rotation))
        # global → ego
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        # ego → sensor (LIDAR_TOP)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        return box

    # GT boxes: global → sensor (filtered to match training-time GT if a list
    # of ann_tokens was supplied; otherwise show all raw nuScenes annotations).
    gt_iter = sample['anns'] if gt_ann_tokens is None else gt_ann_tokens
    for ann_token in gt_iter:
        ann = nusc.get('sample_annotation', ann_token)
        box = _global_to_sensor(ann['translation'], ann['size'], ann['rotation'])
        box.render(ax, view=np.eye(4), normalize=False,
                   colors=('g', 'g', 'g'), linewidth=2)

    # Predicted boxes: global → sensor
    for record in pred_results['results'].get(sample_token, []):
        if record['detection_score'] < score_thr:
            continue
        box = _global_to_sensor(record['translation'], record['size'], record['rotation'])
        box.render(ax, view=np.eye(4), normalize=False,
                   colors=('b', 'b', 'b'), linewidth=1)

    # Legend (upper right)
    legend_handles = [
        mpatches.Patch(edgecolor='green', facecolor='none', linewidth=1.5, label='GT'),
        mpatches.Patch(edgecolor='blue',  facecolor='none', linewidth=1.5, label='Prediction'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.75, edgecolor='gray')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6-surround camera grid (2x3) with NO whitespace between subplots
# --------------------------------------------------------------------------- #
def _render_surround_grid(nusc, sample_token, boxes_per_cam, out_path,
                          dpi=200, figsize_per_cam=(8.0, 4.5)):
    """
    Lay out 6 cameras in a 2x3 grid with zero gaps between images.

    boxes_per_cam: dict[cam_name] -> list of nuScenes Box objects already in
                   the camera's sensor frame.
    """
    sample = nusc.get('sample', sample_token)

    fig_w = 3 * figsize_per_cam[0]
    fig_h = 2 * figsize_per_cam[1]
    fig, axes = plt.subplots(
        2, 3, figsize=(fig_w, fig_h), dpi=dpi,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
    )
    # Eliminate every margin around the figure as well as between subplots
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0,
                        wspace=0.0, hspace=0.0)

    for idx, cam in enumerate(SURROUND_CAMS):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]

        sample_data_token = sample['data'][cam]
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        data_path = nusc.get_sample_data_path(sample_data_token)

        im = Image.open(data_path)
        ax.imshow(im)

        for box in boxes_per_cam.get(cam, []):
            color = np.array(_get_color(nusc, box.name)) / 255.0
            box.render(ax, view=cam_intrinsic, normalize=True,
                       colors=(color, color, color))

        # Force the axes to span the full image, no padding
        ax.set_xlim(0, im.size[0])
        ax.set_ylim(im.size[1], 0)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.margins(0, 0)

    # NOTE: do NOT use bbox_inches='tight' here — it can re-introduce padding.
    plt.savefig(out_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def _build_pred_boxes_per_cam(nusc, sample_token, pred_results, score_thr):
    """Return {cam: [Box, ...]} for a model's predictions, projected per camera."""
    sample = nusc.get('sample', sample_token)
    raw_pred = [
        Box(record['translation'], record['size'],
            Quaternion(record['rotation']),
            name=record['detection_name'], token='predicted')
        for record in pred_results['results'][sample_token]
        if record['detection_score'] >= score_thr
    ]
    boxes_per_cam = {}
    for cam in SURROUND_CAMS:
        sd_token = sample['data'][cam]
        # Each box gets translated/rotated INTO this camera's frame; we therefore
        # need a fresh deep copy per camera, otherwise consecutive cams would
        # accumulate transforms. Box.copy() handles this.
        cam_pred = [b.copy() for b in raw_pred]
        _, projected, _ = _get_predicted_data(nusc, sd_token, pred_anns=cam_pred)
        boxes_per_cam[cam] = projected
    return boxes_per_cam


def _build_gt_boxes_per_cam(nusc, sample_token, gt_ann_tokens=None):
    """If `gt_ann_tokens` is given, only those annotations are kept per camera."""
    sample = nusc.get('sample', sample_token)
    keep = None if gt_ann_tokens is None else set(gt_ann_tokens)
    boxes_per_cam = {}
    for cam in SURROUND_CAMS:
        sd_token = sample['data'][cam]
        _, gt_boxes, _ = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
        if keep is not None:
            gt_boxes = [b for b in gt_boxes if b.token in keep]
        boxes_per_cam[cam] = gt_boxes
    return boxes_per_cam


# --------------------------------------------------------------------------- #
# Per-sample driver: 3 camera files + 2 BEV files
# --------------------------------------------------------------------------- #
def render_sample(nusc, sample_token, baseline_results, ours_results, out_dir,
                  score_thr=0.2, dpi=200, gt_infos=None,
                  gt_classes=NUSC_DET_CLASSES, pc_range=DEFAULT_PC_RANGE):
    os.makedirs(out_dir, exist_ok=True)
    base_path = os.path.join(out_dir, sample_token)

    # Replicate training-time GT filtering: 10 detection classes + pc_range +
    # valid_flag. Only applied when an info pkl was loaded.
    gt_ann_tokens = None
    if gt_infos is not None:
        info = gt_infos.get(sample_token)
        if info is None:
            raise KeyError(f"sample_token {sample_token} not present in gt_pkl")
        sample = nusc.get('sample', sample_token)
        gt_ann_tokens = _filtered_ann_tokens(sample, info, gt_classes, pc_range)

    # --- Camera (3 separate files, 2x3 surround, zero whitespace) ---
    base_per_cam = _build_pred_boxes_per_cam(
        nusc, sample_token, baseline_results, score_thr=score_thr)
    ours_per_cam = _build_pred_boxes_per_cam(
        nusc, sample_token, ours_results, score_thr=score_thr)
    gt_per_cam = _build_gt_boxes_per_cam(nusc, sample_token,
                                         gt_ann_tokens=gt_ann_tokens)

    _render_surround_grid(nusc, sample_token, base_per_cam,
                          out_path=base_path + '_camera_baseline.png', dpi=dpi)
    _render_surround_grid(nusc, sample_token, ours_per_cam,
                          out_path=base_path + '_camera_ours.png', dpi=dpi)
    _render_surround_grid(nusc, sample_token, gt_per_cam,
                          out_path=base_path + '_camera_gt.png', dpi=dpi)

    # --- BEV (one per model: GT green vs that model's pred blue) ---
    render_bev(nusc, sample_token, baseline_results,
               out_path=base_path + '_bev_baseline.png',
               score_thr=score_thr, dpi=dpi, gt_ann_tokens=gt_ann_tokens)
    render_bev(nusc, sample_token, ours_results,
               out_path=base_path + '_bev_ours.png',
               score_thr=score_thr, dpi=dpi, gt_ann_tokens=gt_ann_tokens)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Detection visualization (baseline vs ours).")
    p.add_argument('--results_baseline', required=True,
                   help='Path to baseline detection results JSON (results_nusc.json).')
    p.add_argument('--results_ours', required=True,
                   help='Path to ours detection results JSON.')
    p.add_argument('--out_dir', default='visual_results_v2',
                   help='Directory under which figures are written.')
    p.add_argument('--data_root', default='../data/nuscenes',
                   help='nuScenes dataroot.')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--num_samples', type=int, default=20,
                   help='How many samples (in JSON order) to visualize.')
    p.add_argument('--start_idx', type=int, default=0)
    p.add_argument('--score_thr', type=float, default=0.2,
                   help='Score threshold for drawing predicted boxes on cameras.')
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--sample_tokens', nargs='*', default=None,
                   help='Optional explicit list of sample tokens; overrides start/num.')
    p.add_argument('--scene_tokens', nargs='*', default=None,
                   help='Optional list of scene tokens. If given, only samples '
                        'belonging to these scenes are visualized and '
                        '--scene_keywords is ignored.')
    p.add_argument('--scene_keywords', nargs='*', default=None,
                   help='Filter samples whose scene description (case-insensitive) '
                        'contains any/all of these keywords. e.g. "rain night". '
                        'Ignored if --scene_tokens is provided.')
    p.add_argument('--scene_match', choices=['any', 'all'], default='any',
                   help='Whether the scene description must contain ANY (default) '
                        'or ALL of --scene_keywords.')
    p.add_argument('--list_scenes', action='store_true',
                   help='Print matched (scene_token, sample_token, description) and exit.')
    p.add_argument('--gt_pkl', default=None,
                   help='Path to nuScenes info pkl (e.g. '
                        'nuscenes_infos_temporal_val.pkl). If provided, GT '
                        'panels are filtered to match training-time GT '
                        '(valid_flag, 10 detection classes, point_cloud_range).')
    p.add_argument('--pc_range', nargs=6, type=float, default=list(DEFAULT_PC_RANGE),
                   metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                   help='Point cloud range for GT range-filter (LIDAR frame). '
                        'Default matches projects/configs/diff_bevformer/layout_tiny*.py.')
    return p.parse_args()


def _filter_by_scene(nusc, tokens, keywords, match='any'):
    """Keep only sample tokens whose parent scene.description matches keywords."""
    if not keywords:
        return tokens
    kws = [k.lower() for k in keywords]
    kept = []
    for tok in tokens:
        try:
            sample = nusc.get('sample', tok)
            scene = nusc.get('scene', sample['scene_token'])
        except Exception:
            continue
        desc = (scene.get('description') or '').lower()
        if match == 'all':
            ok = all(k in desc for k in kws)
        else:
            ok = any(k in desc for k in kws)
        if ok:
            kept.append(tok)
    return kept


def _filter_by_scene_tokens(nusc, tokens, scene_tokens):
    """Keep only sample tokens whose parent scene_token is in `scene_tokens`."""
    if not scene_tokens:
        return tokens
    wanted = set(scene_tokens)
    kept = []
    for tok in tokens:
        try:
            sample = nusc.get('sample', tok)
        except Exception:
            continue
        if sample['scene_token'] in wanted:
            kept.append(tok)
    return kept


def main():
    args = parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    baseline_results = mmcv.load(args.results_baseline)
    ours_results = mmcv.load(args.results_ours)

    gt_infos = None
    if args.gt_pkl:
        gt_infos = _load_gt_infos(args.gt_pkl)
        print(f"[gt_pkl] loaded {len(gt_infos)} sample infos from {args.gt_pkl}")
        print(f"[gt_pkl] filtering GT by valid_flag + {len(NUSC_DET_CLASSES)} classes"
              f" + pc_range={tuple(args.pc_range)}")
    else:
        print("[gt_pkl] NOT provided -> GT panels show RAW nuScenes annotations "
              "(may include classes/range/invalid boxes the model was not trained on).")

    # Use the intersection of sample tokens to be robust to JSON differences
    base_tokens = list(baseline_results['results'].keys())
    ours_tokens = set(ours_results['results'].keys())
    aligned = [t for t in base_tokens if t in ours_tokens]
    if not aligned:
        raise RuntimeError("No overlapping sample tokens between the two JSONs.")

    if args.sample_tokens:
        candidates = [t for t in args.sample_tokens if t in aligned]
        if not candidates:
            raise RuntimeError("None of the requested sample_tokens are in both JSONs.")
    else:
        candidates = aligned

    if args.scene_tokens:
        before = len(candidates)
        candidates = _filter_by_scene_tokens(
            nusc, candidates, args.scene_tokens)
        print(f"scene_token filter: {len(args.scene_tokens)} scene(s) -> "
              f"{len(candidates)}/{before} samples")
        if not candidates:
            raise RuntimeError("No samples belong to the requested scene_tokens.")
    elif args.scene_keywords:
        before = len(candidates)
        candidates = _filter_by_scene(
            nusc, candidates, args.scene_keywords, match=args.scene_match)
        print(f"scene filter [{args.scene_match}]: "
              f"{args.scene_keywords} -> {len(candidates)}/{before} samples")
        if not candidates:
            raise RuntimeError("No samples match the requested scene keywords.")

    if args.list_scenes:
        for t in candidates:
            sample = nusc.get('sample', t)
            scene = nusc.get('scene', sample['scene_token'])
            print(f"{scene['token']}\t{t}\t{scene.get('description', '')}")
        return

    if args.sample_tokens:
        targets = candidates
    else:
        s, n = args.start_idx, args.num_samples
        targets = candidates[s:s + n]

    print(f"green = GT,  blue = predictions  (BEV panels)")
    print(f"camera files: per-class color (car=orange, truck=tomato, "
          f"pedestrian=blue, bicycle=crimson, ...)")
    print(f"writing {len(targets)} samples to {args.out_dir}/")
    for tok in targets:
        try:
            render_sample(
                nusc, tok,
                baseline_results=baseline_results,
                ours_results=ours_results,
                out_dir=args.out_dir,
                score_thr=args.score_thr,
                dpi=args.dpi,
                gt_infos=gt_infos,
                gt_classes=NUSC_DET_CLASSES,
                pc_range=tuple(args.pc_range),
            )
            print(f"  done: {tok}")
        except Exception as e:
            print(f"  FAILED: {tok} -> {e}")


if __name__ == '__main__':
    main()
