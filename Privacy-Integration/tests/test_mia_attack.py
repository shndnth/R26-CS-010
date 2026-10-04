"""Membership inference attack: signal extraction, classifier and scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from privacy_integration.attack.classifier import evaluate_attack, interpret, train_classifier
from privacy_integration.attack.features import CONFIDENCE, LOSS, N_FEATURES, extract, loss_gap

CEILING = 55.0


def separable_features(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Members with low loss and high confidence, non-members the opposite."""
    rng = np.random.default_rng(seed)
    members = np.column_stack([rng.uniform(0.8, 1.0, n), rng.uniform(0.0, 0.1, n)])
    nonmembers = np.column_stack([rng.uniform(0.4, 0.7, n), rng.uniform(0.5, 1.0, n)])
    return members, nonmembers


def identical_features(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, N_FEATURES)), rng.uniform(0, 1, (n, N_FEATURES))


def fit(members: np.ndarray, nonmembers: np.ndarray, epochs: int = 40) -> nn.Module:
    torch.manual_seed(0)
    features = np.vstack([members, nonmembers])
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    return train_classifier(features, labels, epochs, "cpu")


class TestFeatures:
    def test_extract_returns_confidence_and_loss_per_sample(self):
        model = nn.Linear(4, 1)
        loader = DataLoader(TensorDataset(torch.randn(10, 4), torch.ones(10, 1)), batch_size=4)
        features = extract(model, loader, "cpu")
        assert features.shape == (10, N_FEATURES)
        assert np.all((features[:, CONFIDENCE] >= 0) & (features[:, CONFIDENCE] <= 1))
        assert np.all(features[:, LOSS] >= 0)

    def test_extract_restores_training_mode(self):
        model = nn.Linear(4, 1)
        model.train()
        extract(model, DataLoader(TensorDataset(torch.randn(4, 4), torch.ones(4, 1))), "cpu")
        assert model.training

    def test_loss_gap_is_nonmember_minus_member(self):
        members = np.array([[0.9, 0.1], [0.9, 0.3]])
        nonmembers = np.array([[0.5, 0.6], [0.5, 0.8]])
        assert loss_gap(members, nonmembers) == pytest.approx(0.5)


class TestAttack:
    def test_separable_signal_is_detected(self):
        members, nonmembers = separable_features()
        outcome = evaluate_attack(fit(members, nonmembers), members, nonmembers, "cpu", CEILING)
        assert outcome.auc > 0.95
        assert outcome.success_rate_pct > 90

    def test_no_signal_stays_near_chance(self):
        members, nonmembers = identical_features()
        held_out_members, held_out_nonmembers = identical_features(seed=1)
        outcome = evaluate_attack(fit(members, nonmembers), held_out_members, held_out_nonmembers,
                                  "cpu", CEILING)
        assert 0.4 < outcome.auc < 0.6

    def test_outcome_reports_sizes_and_loss_gap(self):
        members, nonmembers = separable_features(n=50)
        outcome = evaluate_attack(fit(members, nonmembers, epochs=5), members, nonmembers, "cpu", CEILING)
        assert outcome.n_members == outcome.n_nonmembers == 50
        assert outcome.loss_gap == pytest.approx(loss_gap(members, nonmembers))
        assert set(outcome.as_dict()) >= {"attack_auc", "attack_success_rate_pct", "target_loss_gap"}


class TestInterpretation:
    @pytest.mark.parametrize(
        ("rate", "expected"),
        [(50.0, "Near random"), (54.0, "Within target"), (60.0, "Above target"), (80.0, "High attack")],
    )
    def test_bands(self, rate, expected):
        assert interpret(rate, CEILING).startswith(expected)
