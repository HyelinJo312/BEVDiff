"""Checks for paired-scene statistics used by the splat-mode comparison."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (Path(__file__).resolve().parents[1] / 'projects/bevdiffuser'
               / 'analyze_occ_semantic_splat_modes.py')
SPEC = importlib.util.spec_from_file_location('occ_splat_analysis', MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class PairedSplatStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.core_indices = [index for index, item in enumerate(ANALYSIS.CLASSES)
                             if item[0] in ANALYSIS.CORE]

    def test_ratios_and_undefined_denominators(self):
        result = ANALYSIS.ratios([2, 3, 5])
        self.assertAlmostEqual(float(result['precision']), .4)
        self.assertAlmostEqual(float(result['recall']), 2 / 7)
        self.assertAlmostEqual(float(result['iou']), .2)
        for value in ANALYSIS.ratios([0, 0, 0]).values():
            self.assertTrue(np.isnan(value))

    def test_mask_and_fixed_threshold(self):
        score = np.array([[.1, .05], [.8, .5]])
        target = np.array([[True, True], [False, True]])
        valid = np.array([[True, True], [True, False]])
        np.testing.assert_array_equal(ANALYSIS.count_cells(score, target, valid, .1), [1, 1, 1])

    def test_identical_modes_have_zero_paired_uncertainty(self):
        counts = np.tile([1, 2, 3], (3, 2, len(ANALYSIS.CLASSES), 1))
        rows = ANALYSIS.paired_bootstrap(counts, self.core_indices, draws=64, seed=42)
        for row in rows:
            self.assertEqual(row['valid_draws'], 64)
            self.assertEqual(row['delta_iou'], 0)
            self.assertEqual(row['ci_low'], 0)
            self.assertEqual(row['ci_high'], 0)

    def test_fixed_scene_effect_and_seed(self):
        counts = np.empty((3, 2, len(ANALYSIS.CLASSES), 3), dtype=np.int64)
        counts[:, 0] = [2, 2, 2]
        counts[:, 1] = [3, 1, 1]
        first = ANALYSIS.paired_bootstrap(counts, self.core_indices, draws=64, seed=42)
        second = ANALYSIS.paired_bootstrap(counts, self.core_indices, draws=64, seed=42)
        self.assertEqual(first, second)
        primary = first[0]
        self.assertAlmostEqual(primary['nearest_iou'], 1 / 3)
        self.assertAlmostEqual(primary['bilinear_iou'], .6)
        self.assertAlmostEqual(primary['delta_iou'], .6 - 1 / 3)
        self.assertAlmostEqual(primary['ci_low'], primary['delta_iou'])
        self.assertAlmostEqual(primary['ci_high'], primary['delta_iou'])

    def test_empty_classes_are_not_zero_scores(self):
        counts = np.zeros((3, 2, len(ANALYSIS.CLASSES), 3), dtype=np.int64)
        rows = ANALYSIS.paired_bootstrap(counts, self.core_indices, draws=64, seed=42)
        for row in rows:
            self.assertEqual(row['valid_draws'], 0)
            self.assertTrue(np.isnan(row['delta_iou']))
            self.assertTrue(np.isnan(row['ci_low']))
            self.assertTrue(np.isnan(row['ci_high']))


if __name__ == '__main__':
    unittest.main()
