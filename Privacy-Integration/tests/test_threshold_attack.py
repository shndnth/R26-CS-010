"""Loss-threshold attack and TPR at low FPR."""

from __future__ import annotations

import numpy as np
import pytest

from privacy_integration.attack.threshold import loss_threshold_attack, tpr_at_fpr


def test_separated_losses_are_fully_detected():
    outcome = loss_threshold_attack(np.full(200, 0.05), np.full(200, 2.0))
    assert outcome.auc == pytest.approx(1.0)
    assert outcome.tpr_at_fpr[0.01] == pytest.approx(1.0)


def test_identical_distributions_stay_near_random():
    rng = np.random.default_rng(0)
    outcome = loss_threshold_attack(rng.exponential(1.0, 5000), rng.exponential(1.0, 5000))
    assert outcome.auc == pytest.approx(0.5, abs=0.03)
    assert outcome.tpr_at_fpr[0.01] < 0.03


def test_tpr_at_fpr_respects_the_fpr_bound():
    members = np.array([3.0, 2.0, 1.0, 0.0])
    nonmembers = np.array([2.5, -1.0, -2.0, -3.0])
    assert tpr_at_fpr(members, nonmembers, 0.0) == pytest.approx(0.25)
    assert tpr_at_fpr(members, nonmembers, 0.25) == pytest.approx(1.0)


def test_as_dict_reports_random_reference():
    record = loss_threshold_attack(np.ones(10), np.ones(10)).as_dict()
    assert record["random_tpr_at_1pct_fpr"] == 0.01
    assert record["attack"] == "loss_threshold"
