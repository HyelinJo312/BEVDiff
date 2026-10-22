import numpy as np
import pytest
import torch

from BEVFormer.projects.bevdiffuser.analyze.analyze_occ_teacher_pair import (
    class_rows, confusion, feature_stats, gt_footprint, predecessor_indices, stable_seed,
)


def test_predecessors_are_deterministic_and_scene_bounded():
    infos = [dict(scene_token=scene, timestamp=index) for index, scene in
             enumerate(('a', 'a', 'a', 'b', 'b'))]
    assert predecessor_indices(infos, 0) == ([0, 0, 0, 0], 3)
    assert predecessor_indices(infos, 2) == ([0, 0, 1, 2], 1)
    assert predecessor_indices(infos, 4) == ([3, 3, 3, 4], 2)
    interleaved = [dict(scene_token=scene, timestamp=index) for index, scene in
                   enumerate(('a', 'b', 'a', 'b', 'a'))]
    assert predecessor_indices(interleaved, 4) == ([0, 0, 2, 4], 1)
    assert stable_seed('one') == stable_seed('one') != stable_seed('two')


def test_masked_confusion_and_absent_class():
    target = np.array([[[4, 8], [4, 4]]], dtype=np.uint8)
    pred = np.array([[[4, 4], [8, 4]]], dtype=np.uint8)
    mask = np.array([[[True, True], [False, True]]])
    hist = confusion(pred, target, mask)
    assert hist.sum() == 3
    assert hist[4, 4] == 2 and hist[8, 4] == 1
    rows = class_rows(hist, 'v1')
    assert rows[8]['iou'] == 0 and rows[8]['recall'] == 0
    assert rows[1]['iou'] is None and rows[1]['recall'] is None


def test_feature_stats_shapes_and_empty_region():
    clean = torch.zeros(1, 256, 200, 200)
    feature = torch.ones_like(clean)
    stats = feature_stats(feature, clean)
    assert stats['mse'] == pytest.approx(1)
    assert stats['cells'] == 40000
    empty = feature_stats(feature, clean, torch.zeros(200, 200, dtype=torch.bool))
    assert empty['mse'] is None and empty['cells'] == 0
    with pytest.raises(ValueError):
        feature_stats(feature[:, :, :10], clean)


def test_gt_footprint_transposes_native_xy_to_feature_yx():
    volume = np.zeros((2, 3, 2), dtype=np.uint8)
    visible = np.ones_like(volume, dtype=bool)
    volume[0, 2, 1] = 8
    footprint = gt_footprint(volume, visible, 8)
    assert footprint.shape == (3, 2)
    assert footprint[2, 0] and footprint.sum() == 1
