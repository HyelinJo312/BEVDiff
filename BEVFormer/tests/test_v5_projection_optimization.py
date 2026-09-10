"""Focused regressions for v5 cache, boundary splats, and worker thread limits."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import projects.mmdet3d_plugin
from projects.bevdiffuser.data_utils import (
    CustomNuScenesDataset, CustomNuScenesDiffusionDataset_seg_depth_v2,
)
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner


class ProjectionOptimizationTest(unittest.TestCase):
    def model(self):
        return SegBEVAligner(bev_h=8, bev_w=8, num_classes=3, emb_channels=8,
                             channel_mult=(1, 1, 1), pc_range=(0, 0, 0, 8, 8, 10))

    def test_boundary_weight_is_discarded_not_renormalized(self):
        model = self.model()
        counts = model._splat(torch.tensor([[0., 0., 3.], [1.5, 1.5, 3.]]), torch.tensor([1, 2]))
        self.assertEqual(counts[1, 0, 0].item(), .25)
        self.assertEqual(counts[2, 1, 1].item(), 1.)
        self.assertEqual(counts.sum().item(), 1.25)

    def test_cache_reuse_resize_and_checkpoint_contract(self):
        model = self.model()
        labels = torch.ones(1, 1, 4, 4, dtype=torch.long)
        depth = torch.full(labels.shape, 3.)
        metas = [dict(lidar2img=[np.diag([3., 3., 1., 1.])])]
        before = set(model.state_dict())
        model.project_semantics(labels, metas, depth)
        pointer = model._pixel_grid.data_ptr()
        model.project_semantics(labels, metas, depth)
        self.assertEqual(pointer, model._pixel_grid.data_ptr())
        self.assertEqual(before, set(model.state_dict()))
        self.assertNotIn('_pixel_grid', dict(model.named_buffers()))
        model.pixel_stride = 2
        model.project_semantics(labels, metas, depth)
        self.assertEqual(model._pixel_grid.shape, (4, 3))
        model.project_semantics(labels[:, :, :2], metas, depth[:, :, :2])
        self.assertEqual(model._pixel_grid.shape, (2, 3))

    def test_unknown_and_invalid_remain_zero(self):
        model = self.model()
        labels = torch.tensor([[[[0., -1., 3., 1., 1.5]]]])
        depth = torch.tensor([[[[3., 3., 3., float('nan'), 3.]]]])
        result = model.project_semantics(labels, [dict(lidar2img=[np.eye(4)])], depth)
        for value in result:
            self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_thread_limit_precedes_image_pipeline(self):
        dataset = object.__new__(CustomNuScenesDiffusionDataset_seg_depth_v2)
        dataset.opencv_num_threads = 1
        dataset.use_semantic_bev_cache = False
        dataset.use_semantics = dataset.use_depth = False
        original = cv2.getNumThreads()
        try:
            cv2.setNumThreads(4)
            def base_getitem(instance, index):
                self.assertEqual(cv2.getNumThreads(), 1)
                return dict(img_metas=dict(filename=[]))
            with patch.object(CustomNuScenesDataset, '__getitem__', base_getitem):
                dataset[0]
        finally:
            cv2.setNumThreads(original)


if __name__ == '__main__':
    unittest.main()
