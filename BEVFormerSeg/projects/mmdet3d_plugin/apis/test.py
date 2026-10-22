# Evaluation utilities for BEVFormerSeg (BEVFusion-style 7-class labels).
#
# Differences vs apis/test_v1.py (the legacy HDMapNet-style evaluator):
#   * num_map_class 4 -> 7 (background + 6 BEVFusion classes).
#   * Per-class IoU reported under BEVFusion class names instead of
#     ['Divider', 'Pred Crossing', 'Boundary'].
#   * show_seg uses the BEVFusion palette (matches visual_det_seg_v2.py)
#     and takes a single-channel index map, not one-hot stacks.
#   * Removed hard-coded car-icon path that crashed v1 on this box.

import json
import os.path as osp
import shutil
import tempfile
import time

import cv2
import mmcv
import numpy as np
import torch
import torch.distributed as dist

from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger

from projects.mmdet3d_plugin.metrics import IntersectionOverUnion


# -----------------------------------------------------------------------------
# Class layout — must match bevformer_tiny_seg_v2.py and visual_det_seg_v2.py.
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
NUM_MAP_CLASS = len(SEG_INDEX_TO_NAME)

# BGR palette (cv2). Background = black; other classes follow BEVFusion.
MAP_PALETTE_BGR = {
    'background':    (  0,   0,   0),
    'drivable_area': (227, 206, 166),
    'carpark_area':  (  0, 127, 255),
    'walkway':       ( 28,  26, 227),
    'stop_line':     (111, 191, 253),
    'ped_crossing':  (153, 154, 251),
    'divider':       (154,  61, 106),
}

# Per-class names for the IoU pretty-table (excluding background).
SEG_CLASS_NAMES = [SEG_INDEX_TO_NAME[i] for i in range(1, NUM_MAP_CLASS)]


# -----------------------------------------------------------------------------
def show_seg(semantic_indices):
    """Colorize a (H, W) int index map into a (H, W, 3) BGR image.

    `semantic_indices` may be a torch tensor or a numpy array of shape
    (H, W) (or (1, H, W) which will be squeezed). Returns a uint8 numpy array.
    """
    if isinstance(semantic_indices, torch.Tensor):
        semantic_indices = semantic_indices.detach().cpu().numpy()
    semantic_indices = np.asarray(semantic_indices)
    if semantic_indices.ndim == 3:
        semantic_indices = semantic_indices[0]
    assert semantic_indices.ndim == 2, \
        f'expected (H, W), got {semantic_indices.shape}'

    h, w = semantic_indices.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, name in SEG_INDEX_TO_NAME.items():
        img[semantic_indices == idx] = MAP_PALETTE_BGR[name]
    # Match the original viewer's orientation (flip vertically).
    img = np.flip(img, axis=0).copy()
    return img


def _pred_seg_indices(seg_preds):
    """Convert head logits -> (B, H, W) int64 index map via argmax."""
    return seg_preds.argmax(dim=1).int()


def _pretty_iou_table(num_samples, scores):
    """Build a PrettyTable with per-class IoU and mIoU (excluding bg).

    scores: torch.Tensor of length NUM_MAP_CLASS.
    Returns (PrettyTable, dict, mIoU_float).
    """
    import prettytable as pt
    per_class = [float(scores[i].cpu()) for i in range(1, NUM_MAP_CLASS)]
    miou = float(sum(per_class) / max(1, len(per_class)))
    tb = pt.PrettyTable()
    tb.field_names = ['Validation num'] + SEG_CLASS_NAMES + ['mIoU']
    tb.add_row([num_samples] + [round(v, 4) for v in per_class] + [round(miou, 4)])

    metric_dict = dict(
        Validation_num=num_samples,
        mIoU=round(miou, 4),
        **{name: round(v, 4) for name, v in zip(SEG_CLASS_NAMES, per_class)},
    )
    return tb, metric_dict, miou


# -----------------------------------------------------------------------------
def single_gpu_test(model, data_loader, show=False, out_dir=None,
                    show_score_thr=0.3):
    """Single-GPU test loop that accumulates 7-class semantic-map IoU."""
    logger = get_root_logger()

    test_mode = data_loader.dataset.test_submission
    map_enable = not test_mode
    if map_enable:
        iou_metric = IntersectionOverUnion(NUM_MAP_CLASS).cuda()

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            in_data = {k: v for k, v in data.items() if 'img' in k}
            result = model(return_loss=False, rescale=True, **in_data)

        batch_size = len(result)
        assert batch_size == 1, 'val step batch size must be 1!'

        seg_preds = result[0].get('seg_preds', None)
        pts_bbox  = result[0].get('pts_bbox',  None)

        # ---- optional visualization ----
        if seg_preds is not None and (show or out_dir):
            pred_idx = _pred_seg_indices(seg_preds).cpu().numpy()  # (1, H, W)
            sample_id = data['img_metas'][0].data[0][0]['sample_idx']
            if out_dir:
                mmcv.mkdir_or_exist(out_dir)
                imname = osp.join(out_dir, f'{sample_id}.png')
                cv2.imwrite(imname, show_seg(pred_idx))
                logger.info(f'saving: {imname}')

                # also dump GT for side-by-side inspection if available
                if 'semantic_indices' in data:
                    gt_idx = data['semantic_indices'][0].cpu().numpy()
                    gt_name = osp.join(out_dir, f'{sample_id}_gt.png')
                    cv2.imwrite(gt_name, show_seg(gt_idx))
                    logger.info(f'saving: {gt_name}')

        # ---- accumulate detection results ----
        if pts_bbox is not None:
            results.append(dict(pts_bbox=pts_bbox))

        # ---- accumulate seg IoU ----
        if seg_preds is None:
            map_enable = False
        if map_enable:
            pred_idx = _pred_seg_indices(seg_preds)               # (1, H, W)
            tgt_idx  = data['semantic_indices'][0].unsqueeze(0).cuda()
            iou_metric(pred_idx, tgt_idx)

        for _ in range(batch_size):
            prog_bar.update()

    if map_enable:
        scores = iou_metric.compute()
        tb, metric_dict, _ = _pretty_iou_table(len(dataset), scores)
        print('\n')
        print(tb)
        logger.info(tb)
        with open('segmentation_result.json', 'a') as f:
            f.write(json.dumps(metric_dict) + '\n')

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Multi-GPU test loop with the same accumulator as single_gpu_test."""
    logger = get_root_logger()

    test_mode = data_loader.dataset.test_submission
    map_enable = not test_mode
    if map_enable:
        iou_metric = IntersectionOverUnion(NUM_MAP_CLASS).cuda()

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            in_data = {k: v for k, v in data.items() if 'img' in k}
            result = model(return_loss=False, rescale=True, **in_data)

        batch_size = len(result)
        seg_preds = result[0].get('seg_preds', None)
        pts_bbox  = result[0].get('pts_bbox',  None)

        if pts_bbox is not None:
            results.append(dict(pts_bbox=pts_bbox))

        if seg_preds is None:
            map_enable = False
        if map_enable:
            pred_idx = _pred_seg_indices(seg_preds)
            tgt_idx  = data['semantic_indices'][0].unsqueeze(0).cuda()
            iou_metric(pred_idx, tgt_idx)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if map_enable:
        scores = iou_metric.compute()
        if rank == 0:
            tb, metric_dict, _ = _pretty_iou_table(len(dataset), scores)
            print('\n')
            logger.info(tb)
            with open('segmentation_result.json', 'a') as f:
                f.write(json.dumps(metric_dict) + '\n')

    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


# -----------------------------------------------------------------------------
# Distributed result collection — kept identical to apis/test.py (v1).
# -----------------------------------------------------------------------------
def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    if rank != 0:
        return None
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    ordered_results = []
    for res in part_list:
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    return collect_results_cpu(result_part, size)
