import os.path as osp
from glob import glob

import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from .nuscenes_dataset import CustomNuScenesDataset


OCC3D_CLASS_NAMES = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]


@DATASETS.register_module()
class Occ3DNuScenesDataset(CustomNuScenesDataset):
    """CustomNuScenesDataset wrapper for occupancy-only training.

    BEVFormer/tools/train.py passes detection-oriented default_args while
    building datasets. This wrapper accepts those keys so the stock train
    script can be reused without touching existing files.
    """

    def __init__(self,
                 pc_range=None,
                 use_3d_bbox=None,
                 num_classes=None,
                 num_bboxes=None,
                 occ_root=None,
                 occ_class_names=None,
                 eval_class_indices=None,
                 samples_per_gpu=None,
                 **kwargs):
        self.occ_pc_range = pc_range
        self.occ_num_classes = num_classes or len(OCC3D_CLASS_NAMES)
        self.occ_root = occ_root
        self.occ_class_names = occ_class_names or OCC3D_CLASS_NAMES
        if eval_class_indices is None:
            # Occ3D-nuScenes protocol: mIoU averages the semantic classes
            # only. 'free' marks empty space, dominates the voxel count and
            # is excluded, otherwise the reported mIoU is not comparable to
            # published Occ3D numbers.
            eval_class_indices = [
                index for index, name in enumerate(self.occ_class_names)
                if name != 'free' and index < self.occ_num_classes
            ]
        self.eval_class_indices = eval_class_indices
        self._occ_label_paths = None
        # Index of the frame whose occupancy labels survive union2one.
        self._occ_current_index = None
        super(Occ3DNuScenesDataset, self).__init__(**kwargs)

    def prepare_train_data(self, index):
        """Tag the current frame so history frames skip label loading."""
        self._occ_current_index = index
        try:
            return super(Occ3DNuScenesDataset, self).prepare_train_data(index)
        finally:
            self._occ_current_index = None

    def get_data_info(self, index):
        input_dict = super(Occ3DNuScenesDataset, self).get_data_info(index)
        if input_dict is not None:
            input_dict['load_occ_annotations'] = (
                self._occ_current_index is None
                or index == self._occ_current_index)
        return input_dict

    def _build_occ_label_index(self):
        if self.occ_root is None:
            raise ValueError('occ_root must be set for Occ3D evaluation.')

        label_paths = {}
        pattern = osp.join(self.occ_root, '*', '*', 'labels.npz')
        for path in glob(pattern):
            frame_token = osp.basename(osp.dirname(path))
            label_paths[frame_token] = path
        if len(label_paths) == 0:
            raise FileNotFoundError(
                f'No Occ3D labels.npz files found under {self.occ_root}')
        self._occ_label_paths = label_paths

    def _get_occ_label_path(self, index):
        if self._occ_label_paths is None:
            self._build_occ_label_index()

        info = self.data_infos[index]
        token = info.get('token', info.get('sample_token', None))
        if token is None:
            raise KeyError(
                'Cannot find sample token in data_infos for Occ3D eval.')
        if token not in self._occ_label_paths:
            raise FileNotFoundError(
                f'Cannot find Occ3D labels.npz for sample token {token}')
        return self._occ_label_paths[token]

    def _extract_occ_pred(self, result):
        if isinstance(result, dict):
            if 'occ_pred' not in result:
                raise KeyError('Occ3D result dict must contain "occ_pred".')
            result = result['occ_pred']
        return np.asarray(result)

    def _fast_hist(self, pred, target, mask):
        pred = pred.reshape(-1).astype(np.int64)
        target = target.reshape(-1).astype(np.int64)
        mask = mask.reshape(-1).astype(bool)
        valid = ((target >= 0) & (target < self.occ_num_classes)
                 & (pred >= 0) & (pred < self.occ_num_classes) & mask)
        inds = self.occ_num_classes * target[valid] + pred[valid]
        return np.bincount(
            inds,
            minlength=self.occ_num_classes ** 2).reshape(
                self.occ_num_classes, self.occ_num_classes)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate Occ3D semantic occupancy mIoU under mask_camera."""
        if results is None:
            return {}
        if len(results) == 0:
            raise ValueError('Occ3D evaluation received empty results.')
        if len(results) > len(self):
            raise ValueError(
                f'Got {len(results)} results for dataset of length {len(self)}')

        hist = np.zeros((self.occ_num_classes, self.occ_num_classes),
                        dtype=np.int64)
        for index, result in enumerate(results):
            pred = self._extract_occ_pred(result)
            label = np.load(self._get_occ_label_path(index))
            target = label['semantics']
            mask_camera = label['mask_camera'].astype(bool)

            if pred.shape != target.shape:
                raise ValueError(
                    f'Occ3D prediction shape {pred.shape} does not match '
                    f'target shape {target.shape} at index {index}.')
            hist += self._fast_hist(pred, target, mask_camera)

        tp = np.diag(hist).astype(np.float64)
        gt_count = hist.sum(axis=1).astype(np.float64)
        pred_count = hist.sum(axis=0).astype(np.float64)
        union = gt_count + pred_count - tp
        ious = np.divide(
            tp,
            union,
            out=np.full_like(tp, np.nan, dtype=np.float64),
            where=union > 0)

        class_indices = self.eval_class_indices
        if class_indices is None:
            class_indices = list(range(self.occ_num_classes))
        miou = float(np.nanmean(ious[class_indices]) * 100.0)

        mmcv.print_log('Occ3D validation IoU under mask_camera:', logger)
        for class_idx in class_indices:
            class_name = self.occ_class_names[class_idx]
            class_iou = ious[class_idx]
            if np.isnan(class_iou):
                iou_text = 'nan'
            else:
                iou_text = f'{class_iou * 100.0:.2f}'
            mmcv.print_log(f'{class_name}: {iou_text}', logger)

        metrics = dict()
        metrics['occ3d/mIoU'] = miou
        metrics['occ3d/eval_voxels'] = int(hist.sum())
        for class_idx in class_indices:
            class_name = self.occ_class_names[class_idx]
            metrics[f'occ3d/IoU_{class_name}'] = (
                float(ious[class_idx] * 100.0)
                if not np.isnan(ious[class_idx]) else float('nan'))
        return metrics
