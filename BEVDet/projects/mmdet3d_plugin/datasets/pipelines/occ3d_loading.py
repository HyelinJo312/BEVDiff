import os
from glob import glob

import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadOcc3DAnnotations(object):
    """Load Occ3D-nuScenes voxel labels by nuScenes sample token.

    Expected layout:
        occ_root/scene-name/sample-token/labels.npz

    The loaded voxel arrays keep the Occ3D native layout [X, Y, Z].
    """

    def __init__(self,
                 occ_root,
                 load_lidar_mask=False,
                 strict=True):
        self.occ_root = occ_root
        self.load_lidar_mask = load_lidar_mask
        self.strict = strict
        self.token_to_label_path = self._build_token_index(occ_root)

    @staticmethod
    def _build_token_index(occ_root):
        label_paths = glob(os.path.join(occ_root, '*', '*', 'labels.npz'))
        token_to_label_path = {}
        for label_path in label_paths:
            token = os.path.basename(os.path.dirname(label_path))
            token_to_label_path[token] = label_path
        return token_to_label_path

    def _set_empty(self, results):
        results['gt_occ_semantics'] = None
        results['gt_occ_mask_camera'] = None
        if self.load_lidar_mask:
            results['gt_occ_mask_lidar'] = None
        return results

    def __call__(self, results):
        # History frames of the temporal queue only contribute their images:
        # union2one keeps the annotations of the last frame only, so decoding
        # their labels.npz would be thrown away.
        if not results.get('load_occ_annotations', True):
            return self._set_empty(results)

        sample_token = results['sample_idx']
        label_path = self.token_to_label_path.get(sample_token)

        if label_path is None:
            if self.strict:
                raise FileNotFoundError(
                    f'Occ3D label not found for sample token {sample_token} '
                    f'under {self.occ_root}')
            return self._set_empty(results)

        occ = np.load(label_path)
        results['gt_occ_semantics'] = occ['semantics'].astype(np.int64)
        results['gt_occ_mask_camera'] = occ['mask_camera'].astype(np.bool_)
        if self.load_lidar_mask:
            results['gt_occ_mask_lidar'] = occ['mask_lidar'].astype(np.bool_)
        results['occ_label_path'] = label_path
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'occ_root={self.occ_root}, '
                f'load_lidar_mask={self.load_lidar_mask}, '
                f'strict={self.strict}, '
                f'num_labels={len(self.token_to_label_path)})')
