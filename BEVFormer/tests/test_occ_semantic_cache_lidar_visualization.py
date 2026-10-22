"""Small synthetic checks for ego-frame cache/LiDAR visualization."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'projects/bevdiffuser/visualize_occ_semantic_cache_lidar.py'
SPEC = importlib.util.spec_from_file_location('occ_cache_lidar_visualization', SCRIPT)
VISUALIZATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISUALIZATION)


class CacheLidarVisualizationTest(unittest.TestCase):
    def test_legacy_box_corners_match_mmdet3d(self):
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        box = np.array([1., 2., 3., 2., 4., 1.5, .3], dtype=np.float32)
        actual = VISUALIZATION.lidar_box_corners(box)
        reference = LiDARInstance3DBoxes(box[None], origin=(.5, .5, .5)).corners.numpy()[0]
        distance = np.linalg.norm(actual[:, None] - reference[None], axis=-1)
        self.assertLess(float(distance.min(axis=1).max()), 1e-5)
        self.assertLess(float(distance.min(axis=0).max()), 1e-5)

    def test_gt_boxes_transform_all_corners_and_filter_invalid(self):
        info = dict(gt_boxes=np.array([[1., 0., 0., 2., 4., 2., 0.],
                                       [1., 0., 0., 2., 4., 2., 0.],
                                       [100., 0., 0., 2., 4., 2., 0.]]),
                    gt_names=['car', 'truck', 'car'], valid_flag=[True, False, True],
                    lidar2ego_rotation=[np.sqrt(.5), 0., 0., np.sqrt(.5)],
                    lidar2ego_translation=[1., 2., 3.])
        boxes = VISUALIZATION.gt_boxes_ego(info, np.array([-10., -10., -10., 10., 10., 10.]))
        self.assertEqual(len(boxes), 1)
        corners = np.array(boxes[0]['corners_ego'])
        np.testing.assert_allclose(corners.mean(axis=0), [1., 3., 3.], atol=1e-12)
        np.testing.assert_allclose(np.ptp(corners, axis=0), [4., 2., 2.], atol=1e-12)
        self.assertEqual(boxes[0]['name'], 'car')

    def test_white_lidar_background_and_gt_on_all_panels(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        record = dict(gt_boxes_in_range=1, observed_percent=25., split='train',
                      scene_token='test-scene', token='sample-token',
                      gt_boxes=[dict(name='car', polygon_xy=[[-1., -2.], [1., -2.],
                                                            [1., 2.], [-1., 2.]])])
        figure, axes = plt.subplots(1, 3)
        try:
            VISUALIZATION.draw_sample(axes, record, np.array([[1., 2., .5]]),
                                      np.zeros((50, 50), dtype=int), [-40., 40., -40., 40.],
                                      np.array([-40., -40., -1., 40., 40., 5.4]), plt)
            np.testing.assert_allclose(axes[0].get_facecolor(), [1., 1., 1., 1.])
            self.assertTrue(all(len(axis.patches) == 1 for axis in axes))
        finally:
            plt.close(figure)

    def test_lidar_to_ego_rotation_then_translation(self):
        points = np.array([[1., 0., 0.], [0., 1., 1.]])
        rotation = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        actual = VISUALIZATION.lidar_to_ego(points, rotation, [1., 2., 3.])
        np.testing.assert_allclose(actual, [[1., 3., 3.], [0., 2., 4.]], atol=1e-12)
        np.testing.assert_array_equal(points, [[1., 0., 0.], [0., 1., 1.]])

    def test_argmax_preserves_grid_and_unknown_cells(self):
        probability = np.zeros((17, 3, 4), dtype=np.float32)
        probability[1, 1, 2] = .01
        probability[15, 1, 2] = .02
        before = probability.copy()
        labels, observed = VISUALIZATION.dominant_labels(probability)
        self.assertEqual(labels.shape, (3, 4))
        self.assertEqual(labels[1, 2], 15)
        self.assertEqual(int(observed.sum()), 1)
        self.assertEqual(labels[0, 0], 0)
        np.testing.assert_array_equal(probability, before)

    def test_invalid_probabilities_are_rejected(self):
        for value in (-1., np.nan, np.inf):
            probability = np.zeros((17, 3, 4), dtype=np.float32)
            probability[1, 0, 0] = value
            with self.assertRaises(ValueError):
                VISUALIZATION.dominant_labels(probability)

    def test_seeded_selection_uses_distinct_scenes_and_middle_cached_frame(self):
        infos = [dict(token=f'{scene}-{frame}', scene_token=str(scene), timestamp=frame)
                 for scene in range(5) for frame in range(3)]
        entries = {info['token']: {} for info in infos}
        first = VISUALIZATION.select_frames(infos, entries, 3, np.random.default_rng(42))
        second = VISUALIZATION.select_frames(infos, entries, 3, np.random.default_rng(42))
        self.assertEqual(first, second)
        self.assertEqual(len({info['scene_token'] for info in first}), 3)
        self.assertTrue(all(info['timestamp'] == 1 for info in first))
        with self.assertRaises(ValueError):
            VISUALIZATION.select_frames(infos, {}, 3, np.random.default_rng(42))


if __name__ == '__main__':
    unittest.main()
