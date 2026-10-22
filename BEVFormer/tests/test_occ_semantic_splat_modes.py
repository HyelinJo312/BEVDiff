"""Focused statistics checks for the paired occupancy splat analysis."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'projects/bevdiffuser/analyze_occ_semantic_splat_modes.py'
SPEC = importlib.util.spec_from_file_location('analyze_occ_semantic_splat_modes', SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class SplatModeStatisticsTest(unittest.TestCase):
    def test_pooled_counts_not_mean_frame_iou(self):
        frame_counts = np.array([[1, 0, 0], [0, 8, 1]])
        self.assertAlmostEqual(float(ANALYSIS.ratios(frame_counts.sum(axis=0))['iou']), .1)
        frame_iou = ANALYSIS.ratios(frame_counts)['iou']
        self.assertAlmostEqual(float(frame_iou.mean()), .5)

    def test_undefined_denominators_remain_nan(self):
        score = ANALYSIS.ratios(np.array([0, 0, 0]))
        self.assertTrue(all(np.isnan(value) for value in score.values()))
        score = ANALYSIS.ratios(np.array([0, 2, 0]))
        self.assertEqual(float(score['precision']), 0)
        self.assertEqual(float(score['iou']), 0)
        self.assertTrue(np.isnan(score['recall']))

    def test_gt_negative_false_positives_are_counted(self):
        prediction_score = np.array([[.11, .2], [0, 0]])
        target = np.zeros((2, 2), dtype=bool)
        valid = np.ones_like(target)
        np.testing.assert_array_equal(ANALYSIS.count_cells(prediction_score, target, valid, .1),
                                      np.array([0, 2, 0]))

    def test_identical_modes_have_zero_paired_delta(self):
        frame_counts = np.zeros((4, 2, len(ANALYSIS.CLASSES), 3), dtype=np.int64)
        frame_counts[:, :, :, :] = (2, 3, 4)
        core_indices = [index for index, item in enumerate(ANALYSIS.CLASSES)
                        if item[0] in ANALYSIS.CORE]
        rows = ANALYSIS.paired_bootstrap(frame_counts, core_indices, draws=80, seed=9)
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row['delta_iou'], 0)
            self.assertEqual(row['ci_low'], 0)
            self.assertEqual(row['ci_high'], 0)
            self.assertEqual(row['valid_draws'], 80)

    def test_sparse_draws_omit_zero_denominator_and_seed_repeats(self):
        frame_counts = np.zeros((3, 2, len(ANALYSIS.CLASSES), 3), dtype=np.int64)
        frame_counts[0, :, :, :] = (1, 0, 0)
        frame_counts[1, :, :, :] = (0, 1, 0)
        frame_counts[2, :, :, :] = (0, 0, 0)
        core_indices = [index for index, item in enumerate(ANALYSIS.CLASSES)
                        if item[0] in ANALYSIS.CORE]
        first = ANALYSIS.paired_bootstrap(frame_counts, core_indices, draws=250, seed=20261015)
        second = ANALYSIS.paired_bootstrap(frame_counts, core_indices, draws=250, seed=20261015)
        self.assertEqual(first, second)
        self.assertGreater(first[0]['valid_draws'], 0)
        self.assertLess(first[0]['valid_draws'], 250)
        self.assertEqual(first[0]['delta_iou'], 0)

    def test_synthetic_figures(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        valid = np.ones((50, 50), dtype=bool)
        targets = [np.zeros((50, 50), dtype=bool) for _ in ANALYSIS.CLASSES]
        probabilities = {mode: np.zeros((17, 50, 50), dtype=np.float32)
                         for mode in ANALYSIS.MODES}
        frame_counts = np.zeros((2, 2, len(ANALYSIS.CLASSES), 3), dtype=np.int64)
        frame_counts[:] = (1, 1, 1)
        core_indices = [index for index, item in enumerate(ANALYSIS.CLASSES)
                        if item[0] in ANALYSIS.CORE]
        bootstrap = ANALYSIS.paired_bootstrap(frame_counts, core_indices, draws=10, seed=3)
        thresholds = [dict(endpoint='macro4', mode=mode, threshold=threshold, iou=.5)
                      for threshold in ANALYSIS.THRESHOLDS for mode in ANALYSIS.MODES]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ANALYSIS.plot_modes(output / 'frame.png', 'synthetic', valid, targets, probabilities, plt)
            ANALYSIS.plot_errors(output / 'errors.png', 'synthetic', valid, targets, probabilities, plt)
            ANALYSIS.plot_summary(output, bootstrap, thresholds, plt)
            for name in ('frame.png', 'errors.png', 'paired_delta_ci.png', 'threshold_sensitivity.png'):
                self.assertGreater((output / name).stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
