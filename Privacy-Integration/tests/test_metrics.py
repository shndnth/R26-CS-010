"""Classification metric computation."""

from __future__ import annotations

import numpy as np
import pytest

from privacy_integration.evaluation.metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_separation(self):
        m = compute_metrics(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0]))
        assert m.accuracy == 100.0
        assert m.precision == 100.0
        assert m.recall == 100.0
        assert m.auc == 1.0

    def test_confusion_matrix_counts(self):
        # predictions: 1, 0, 1, 0  against truth 1, 1, 0, 0
        m = compute_metrics(np.array([0.9, 0.1, 0.7, 0.2]), np.array([1, 1, 0, 0]))
        assert (m.true_positives, m.false_negatives) == (1, 1)
        assert (m.false_positives, m.true_negatives) == (1, 1)

    def test_majority_predictor_scores_high_accuracy_but_zero_recall(self):
        probs = np.full(100, 0.1)
        targets = np.array([1] * 10 + [0] * 90)
        m = compute_metrics(probs, targets)
        assert m.accuracy == 90.0
        assert m.recall == 0.0
        assert m.precision == 0.0

    def test_auc_is_nan_for_single_class_targets(self):
        m = compute_metrics(np.array([0.3, 0.7]), np.array([1, 1]))
        assert np.isnan(m.auc)

    def test_f1_is_harmonic_mean(self):
        m = compute_metrics(np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 1, 0, 0]))
        expected = 2 * (m.precision / 100) * (m.recall / 100) / ((m.precision + m.recall) / 100)
        assert m.f1 / 100 == pytest.approx(expected)

    def test_threshold_shifts_predictions(self):
        probs = np.array([0.6, 0.6, 0.4, 0.4])
        targets = np.array([1, 1, 0, 0])
        assert compute_metrics(probs, targets, threshold=0.5).accuracy == 100.0
        assert compute_metrics(probs, targets, threshold=0.7).recall == 0.0

    def test_auc_is_invariant_to_class_imbalance(self):
        balanced = compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]))
        imbalanced = compute_metrics(
            np.array([0.9] + [0.1] * 9), np.array([1] + [0] * 9)
        )
        assert balanced.auc == imbalanced.auc == 1.0

    def test_as_dict_round_trips_all_fields(self):
        m = compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]), loss=0.25)
        d = m.as_dict()
        assert d["loss"] == 0.25
        assert set(d) >= {"accuracy", "precision", "recall", "f1", "auc"}
