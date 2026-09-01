import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class Occ3DFormatBundle(object):
    """Format Occ3D voxel labels for batched training."""

    def __init__(self,
                 keys=('gt_occ_semantics', 'gt_occ_mask_camera',
                       'gt_occ_mask_lidar')):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            if key not in results or results[key] is None:
                continue
            tensor = to_tensor(results[key])
            if key == 'gt_occ_semantics':
                tensor = tensor.long()
            else:
                tensor = tensor.to(torch.bool)
            results[key] = DC(tensor, stack=True)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'
