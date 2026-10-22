# Visualization for BEVFormerSeg V2 (BEVFusion-style seg labels).
# Renders a BEV semantic map (GT or model prediction) next to the 6 camera views,
# with detection boxes overlaid. Palette follows BEVFusion conventions.
#
# Typical usage (GT only — no prediction available yet):
#   python visual_det_seg_v2.py \
#       --det-results <path>/pts_bbox/results_nusc.json \
#       --out-dir visual_res_v2
#
# With model seg predictions saved as .pkl (output of tools/test.py --out X.pkl):
#   python visual_det_seg_v2.py \
#       --det-results <path>/pts_bbox/results_nusc.json \
#       --seg-results <path>/results.pkl \
#       --ann-file   data/nuscenes/nuscenes_infos_temporal_val.pkl \
#       --mode both

import argparse
import os
import os.path as osp
import pickle

import cv2
import mmcv
import numpy as np
import torch
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from pyquaternion import Quaternion

# Register PIPELINES (also exposes BEVFusionSegmentationV2)
import projects.mmdet3d_plugin  # noqa: F401
from mmdet.datasets.builder import PIPELINES

'''

# GT만 시각화 (학습 전에도 확인 가능)
python visual_det_seg_v2.py \
    --det-results test/.../pts_bbox/results_nusc.json \
    --out-dir visual_res_v2 --mode gt --num 50

# 모델 예측까지 같이 보고 싶을 때
python visual_det_seg_v2.py \
    --det-results test/.../pts_bbox/results_nusc.json \
    --seg-results test/.../results.pkl \
    --ann-file data/nuscenes/nuscenes_infos_temporal_val.pkl \
    --mode both --num 50
    
'''


# -----------------------------------------------------------------------------
# Class layout — must match bevformer_tiny_seg_v2.py
# -----------------------------------------------------------------------------
SEG_INDEX_TO_NAME = {
    0: 'background',
    1: 'drivable_area',
    2: 'carpark_area',
    3: 'walkway',
    4: 'stop_line',
    5: 'ped_crossing',
    6: 'divider',
}
NUM_SEG_CLASSES = len(SEG_INDEX_TO_NAME)

# BEVFusion-style palette in BGR (cv2 convention).
# Reference: BEVFusion/tools/visualize.py MAP_PALETTE (RGB -> BGR here).
MAP_PALETTE_BGR = {
    'background':    (  0,   0,   0),   # black
    'drivable_area': (227, 206, 166),   # light blue
    'carpark_area':  (  0, 127, 255),   # orange
    'walkway':       ( 28,  26, 227),   # red
    'stop_line':     (111, 191, 253),   # peach
    'ped_crossing':  (153, 154, 251),   # pink
    'divider':       (154,  61, 106),   # purple
}

# Must match bevformer_tiny_seg_v2.py
DET_GRID_CONF = {
    'xbound': [-51.2, 51.2, 0.68],
    'ybound': [-51.2, 51.2, 0.68],
}
MAP_GRID_CONF = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
}

CAMS_GRID = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]


# -----------------------------------------------------------------------------
# Seg label generation (GT) and prediction loading
# -----------------------------------------------------------------------------
def build_gt_pipeline(dataset_root):
    return PIPELINES.build(dict(
        type='BEVFusionSegmentationV2',
        dataset_root=dataset_root,
        map_grid_conf=MAP_GRID_CONF,
    ))


def gen_gt_semantic_indices(nusc, sample_token, gt_pipeline):
    """Build the GT semantic-index map for one sample, exactly like training."""
    sample = nusc.get('sample', sample_token)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    log = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])
    results = {
        'location': log['location'],
        'ego2global_translation': pose['translation'],
        'ego2global_rotation':    pose['rotation'],
    }
    out = gt_pipeline(results)
    return out['semantic_indices'].numpy().astype(np.int64)  # (H, W)


def pred_indices_from_entry(entry):
    """Take one result_dict entry and return (H, W) index map (or None)."""
    sp = entry.get('seg_preds', None) if isinstance(entry, dict) else None
    if sp is None:
        return None
    if isinstance(sp, torch.Tensor):
        sp = sp.detach().cpu().numpy()
    sp = np.asarray(sp)
    if sp.ndim == 4:
        sp = sp[0]
    if sp.ndim == 3:        # (C, H, W) logits -> argmax
        return sp.argmax(axis=0).astype(np.int64)
    if sp.ndim == 2:        # already index map
        return sp.astype(np.int64)
    raise ValueError(f'Unexpected seg_preds shape: {sp.shape}')


# -----------------------------------------------------------------------------
# Colorization + padding into the (larger) detection-grid canvas
# -----------------------------------------------------------------------------
def colorize_semantic_indices(si):
    """(H, W) int index map -> (H, W, 3) BGR uint8."""
    h, w = si.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, name in SEG_INDEX_TO_NAME.items():
        img[si == idx] = MAP_PALETTE_BGR[name]
    return img


def padding_seg_to_det(seg_bgr):
    """Place the (small) map-grid canvas centered inside a square det-grid
    canvas, then rotate so ego forward points up — same convention as the
    original viewer. Out-of-map region uses mid-gray so seg classes stand out.
    """
    h, w, _ = seg_bgr.shape
    ratio = (DET_GRID_CONF['xbound'][1] - DET_GRID_CONF['xbound'][0]) / \
            (MAP_GRID_CONF['xbound'][1] - MAP_GRID_CONF['xbound'][0])
    det_w = int(ratio * w)
    det_h = det_w
    canvas = np.full((det_h, det_w, 3), 64, dtype=np.uint8)
    y0 = det_h // 2 - h // 2
    x0 = det_w // 2 - w // 2
    canvas[y0:y0 + h, x0:x0 + w, :] = seg_bgr
    canvas = np.rot90(canvas, 1, [0, 1]).copy()
    return canvas


# -----------------------------------------------------------------------------
# BEV box overlay (GT red, Pred blue) on the colorized seg canvas
# -----------------------------------------------------------------------------
def _bev_view_matrix(canvas_size):
    scale = canvas_size / (DET_GRID_CONF['xbound'][1] * 2)
    return np.array([[scale,      0, 0, canvas_size / 2],
                     [0,     -scale, 0, canvas_size / 2],
                     [0,          0, 1, 0],
                     [0,          0, 0, 1]])


def render_bev_with_boxes(seg_canvas, nusc, sample_token,
                          gt_eval, pred_eval, conf_th=0.30):
    sample_rec = nusc.get('sample', sample_token)
    sd_record  = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record  = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_rec   = nusc.get('ego_pose', sd_record['ego_pose_token'])

    gt_boxes  = boxes_to_sensor(gt_eval[sample_token],  pose_rec, cs_record)
    est_boxes = boxes_to_sensor(pred_eval[sample_token], pose_rec, cs_record)
    for b, bg in zip(est_boxes, pred_eval[sample_token]):
        b.score = bg.detection_score

    view = _bev_view_matrix(seg_canvas.shape[0])
    # GT — red
    for box in gt_boxes:
        box.render_cv2(seg_canvas, view=view,
                       colors=((0, 0, 255),) * 3, linewidth=2)
    # Pred — blue
    for box in est_boxes:
        if np.isnan(box.score) or box.score < conf_th:
            continue
        box.render_cv2(seg_canvas, view=view, normalize=False,
                       colors=((255, 0, 0),) * 3, linewidth=1)
    return seg_canvas


def build_eval_boxes(nusc, sample_token, bevformer_results):
    bbox_gt, bbox_pred = [], []
    for ann in nusc.get('sample', sample_token)['anns']:
        c = nusc.get('sample_annotation', ann)
        try:
            bbox_gt.append(DetectionBox(
                sample_token=c['sample_token'],
                translation=tuple(c['translation']),
                size=tuple(c['size']),
                rotation=tuple(c['rotation']),
                velocity=nusc.box_velocity(c['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0),
                num_pts=-1 if 'num_pts' not in c else int(c['num_pts']),
                detection_name=category_to_detection_name(c['category_name']),
                detection_score=-1.0,
                attribute_name=''))
        except Exception:
            pass
    for c in bevformer_results['results'][sample_token]:
        bbox_pred.append(DetectionBox(
            sample_token=c['sample_token'],
            translation=tuple(c['translation']),
            size=tuple(c['size']),
            rotation=tuple(c['rotation']),
            velocity=tuple(c['velocity']),
            ego_translation=(0.0, 0.0, 0.0),
            num_pts=-1,
            detection_name=c['detection_name'],
            detection_score=float(c.get('detection_score', -1.0)),
            attribute_name=c['attribute_name']))
    gt_eval, pred_eval = EvalBoxes(), EvalBoxes()
    gt_eval.add_boxes(sample_token, bbox_gt)
    pred_eval.add_boxes(sample_token, bbox_pred)
    return gt_eval, pred_eval


# -----------------------------------------------------------------------------
# Camera grid (2 x 3) with predicted boxes
# -----------------------------------------------------------------------------
def get_color_for_cam_box(nusc, category_name):
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    if category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    if category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']
    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return (0, 0, 0)


def _get_predicted_data(nusc, sample_data_token, pred_anns,
                        box_vis_level=BoxVisibility.ANY):
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    data_path = nusc.get_sample_data_path(sample_data_token)

    cam_intrinsic = np.array(cs_record['camera_intrinsic']) \
        if sensor_record['modality'] == 'camera' else None
    imsize = (sd_record['width'], sd_record['height']) \
        if sensor_record['modality'] == 'camera' else None

    box_list = []
    for box in pred_anns:
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        if sensor_record['modality'] == 'camera':
            if not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue
        box_list.append(box)
    return data_path, box_list, cam_intrinsic


def render_cameras(nusc, sample_token, pred_data, score_th=0.30):
    sample = nusc.get('sample', sample_token)
    imgs = []
    for cam in CAMS_GRID:
        sd_token = sample['data'][cam]
        boxes = [Box(r['translation'], r['size'], Quaternion(r['rotation']),
                     name=r['detection_name'], token='predicted')
                 for r in pred_data['results'][sample_token]
                 if r['detection_score'] > score_th]
        data_path, boxes_pred, cam_intrinsic = _get_predicted_data(
            nusc, sd_token, boxes)
        img = cv2.imread(data_path)
        for box in boxes_pred:
            c = get_color_for_cam_box(nusc, box.name)
            box.render_cv2(img, view=cam_intrinsic, normalize=True,
                           colors=(c, c, c))
        imgs.append(img)
    row1 = np.hstack(imgs[:3])
    row2 = np.hstack(imgs[3:])
    return np.vstack((row1, row2))


# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------
def make_legend(height, width=260, padding=10):
    legend = np.full((height, width, 3), 32, dtype=np.uint8)
    row_h = max(20, (height - 2 * padding) // (NUM_SEG_CLASSES + 3))
    y = padding
    cv2.putText(legend, 'BEV seg classes', (padding, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += row_h
    for idx in range(NUM_SEG_CLASSES):
        name = SEG_INDEX_TO_NAME[idx]
        color = MAP_PALETTE_BGR[name]
        cv2.rectangle(legend, (padding, y), (padding + 26, y + row_h - 6),
                      color, -1)
        cv2.putText(legend, f'{idx}: {name}',
                    (padding + 36, y + row_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                    cv2.LINE_AA)
        y += row_h
    y += row_h // 2
    cv2.putText(legend, 'GT box',
                (padding, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, 'Pred box',
                (padding, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 0, 0), 1, cv2.LINE_AA)
    return legend


# -----------------------------------------------------------------------------
# Per-sample composition
# -----------------------------------------------------------------------------
def render_sample(nusc, sample_token, bevformer_results,
                  gt_pipeline, pred_entry=None, mode='gt'):
    """mode: 'gt', 'pred', or 'both'."""
    gt_eval, pred_eval = build_eval_boxes(nusc, sample_token, bevformer_results)

    if mode in ('gt', 'both'):
        gt_si = gen_gt_semantic_indices(nusc, sample_token, gt_pipeline)
        gt_canvas = padding_seg_to_det(colorize_semantic_indices(gt_si))
        gt_bev = render_bev_with_boxes(gt_canvas, nusc, sample_token,
                                       gt_eval, pred_eval)
    if mode in ('pred', 'both'):
        pr_si = pred_indices_from_entry(pred_entry) if pred_entry is not None else None
        if pr_si is None:
            if mode == 'pred':
                # Fall back to GT so at least the BEV column is informative.
                gt_si = gen_gt_semantic_indices(nusc, sample_token, gt_pipeline)
                gt_canvas = padding_seg_to_det(colorize_semantic_indices(gt_si))
                gt_bev = render_bev_with_boxes(gt_canvas, nusc, sample_token,
                                               gt_eval, pred_eval)
                mode = 'gt'
            else:
                # 'both' but no pred — fall back to GT-only
                mode = 'gt'
        else:
            pr_canvas = padding_seg_to_det(colorize_semantic_indices(pr_si))
            pr_bev = render_bev_with_boxes(pr_canvas, nusc, sample_token,
                                           gt_eval, pred_eval)

    if mode == 'gt':
        bev = gt_bev
        cv2.putText(bev, 'GT seg', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    elif mode == 'pred':
        bev = pr_bev
        cv2.putText(bev, 'Pred seg', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(gt_bev, 'GT seg',  (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pr_bev, 'Pred seg', (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        bev = np.vstack([gt_bev, pr_bev])

    cam_img = render_cameras(nusc, sample_token, bevformer_results)
    target_h = cam_img.shape[0]
    bev_resized = cv2.resize(
        bev, (int(bev.shape[1] * target_h / bev.shape[0]), target_h),
        interpolation=cv2.INTER_LINEAR)
    legend = make_legend(target_h)
    return np.hstack([legend, bev_resized, cam_img])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default='data/nuscenes',
                   help='nuScenes dataset root')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--det-results', required=True,
                   help='BEVFormer detection results json '
                        '(typically <test>/pts_bbox/results_nusc.json)')
    p.add_argument('--seg-results', default=None,
                   help='Optional: results.pkl from tools/test.py '
                        '(contains seg_preds per sample)')
    p.add_argument('--ann-file', default=None,
                   help='Required with --seg-results: dataset annotation pkl '
                        '(e.g. nuscenes_infos_temporal_val.pkl) used to align '
                        'seg_results list index <-> sample_token')
    p.add_argument('--out-dir', default='visual_res_v2')
    p.add_argument('--mode', default='gt', choices=['gt', 'pred', 'both'])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--num', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading NuScenes ({args.version}) from {args.dataroot} ...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f'Loading detection results: {args.det_results}')
    bevformer_results = mmcv.load(args.det_results)

    dataroot = args.dataroot if args.dataroot.endswith('/') else args.dataroot + '/'
    gt_pipeline = build_gt_pipeline(dataroot)

    sample_to_pred = {}
    if args.seg_results is not None:
        assert args.ann_file is not None, '--ann-file is required with --seg-results'
        with open(args.seg_results, 'rb') as f:
            seg_list = pickle.load(f)
        anns = mmcv.load(args.ann_file)
        infos = sorted(anns['infos'], key=lambda e: e['timestamp'])
        assert len(seg_list) == len(infos), (
            f'seg results length {len(seg_list)} != dataset length {len(infos)}')
        for info, res in zip(infos, seg_list):
            sample_to_pred[info['token']] = res

    sample_token_list = list(bevformer_results['results'].keys())
    end = min(args.start + args.num, len(sample_token_list))
    print(f'Rendering samples [{args.start}, {end}) -> {args.out_dir}')

    for i in range(args.start, end):
        tok = sample_token_list[i]
        pred_entry = sample_to_pred.get(tok) if args.mode != 'gt' else None
        try:
            img = render_sample(nusc, tok, bevformer_results,
                                gt_pipeline, pred_entry, mode=args.mode)
        except Exception as e:
            print(f'  [{i}] {tok} failed: {e}')
            continue
        out_path = osp.join(args.out_dir, f'{tok}.jpg')
        cv2.imwrite(out_path, img)
        print(f'  [{i}] {out_path}')


if __name__ == '__main__':
    main()
