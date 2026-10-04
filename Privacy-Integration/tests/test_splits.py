"""Stratified splitting, rebalancing and subsampling."""

from __future__ import annotations

import pytest

from privacy_integration.data.splits import (
    DataSplit,
    balance_by_oversampling,
    stratified_split,
    subsample_to,
)

LABELS = [1] * 100 + [0] * 900


class TestDataSplit:
    def test_rejects_overlapping_partitions(self):
        with pytest.raises(ValueError, match="overlap"):
            DataSplit(train=[1, 2], val=[2, 3], test=[4])

    def test_accepts_disjoint_partitions(self):
        assert DataSplit(train=[1], val=[2], test=[3]).sizes == (1, 1, 1)


class TestStratifiedSplit:
    def test_partitions_are_disjoint(self):
        split = stratified_split(LABELS, 0.7, 0.15, seed=42)
        assert not set(split.train) & set(split.val)
        assert not set(split.train) & set(split.test)
        assert not set(split.val) & set(split.test)

    def test_every_index_is_assigned(self):
        split = stratified_split(LABELS, 0.7, 0.15, seed=42)
        assert len(set(split.train) | set(split.val) | set(split.test)) == len(LABELS)

    def test_preserves_class_balance(self):
        split = stratified_split(LABELS, 0.7, 0.15, seed=42)
        for part in (split.train, split.val, split.test):
            rate = sum(LABELS[i] for i in part) / len(part)
            assert rate == pytest.approx(0.10, abs=0.03)

    def test_is_deterministic_for_a_seed(self):
        assert stratified_split(LABELS, 0.7, 0.15, 42) == stratified_split(LABELS, 0.7, 0.15, 42)

    def test_differs_across_seeds(self):
        assert stratified_split(LABELS, 0.7, 0.15, 42) != stratified_split(LABELS, 0.7, 0.15, 43)

    def test_rejects_fractions_leaving_no_test_split(self):
        with pytest.raises(ValueError, match="test split"):
            stratified_split(LABELS, 0.9, 0.15, seed=42)

    def test_rejects_out_of_range_fractions(self):
        with pytest.raises(ValueError):
            stratified_split(LABELS, 1.5, 0.1, seed=42)


class TestBalanceByOversampling:
    def test_equalises_class_counts(self):
        balanced = balance_by_oversampling(range(len(LABELS)), LABELS)
        positives = sum(LABELS[i] for i in balanced)
        assert positives == len(balanced) - positives

    def test_only_duplicates_existing_indices(self):
        balanced = balance_by_oversampling(range(len(LABELS)), LABELS)
        assert set(balanced) <= set(range(len(LABELS)))

    def test_retains_every_original_index(self):
        balanced = balance_by_oversampling(range(len(LABELS)), LABELS)
        assert set(balanced) == set(range(len(LABELS)))

    def test_single_class_input_is_unchanged(self):
        assert balance_by_oversampling([0, 1, 2], [0, 0, 0]) == [0, 1, 2]

    def test_already_balanced_input_is_unchanged_in_size(self):
        labels = [0, 1, 0, 1]
        assert len(balance_by_oversampling(range(4), labels)) == 4


class TestSubsampleTo:
    def test_returns_requested_size(self):
        assert len(subsample_to(range(100), 30, seed=42)) == 30

    def test_returns_all_when_size_exceeds_input(self):
        assert len(subsample_to(range(10), 50, seed=42)) == 10

    def test_draws_without_replacement(self):
        drawn = subsample_to(range(100), 30, seed=42)
        assert len(set(drawn)) == 30

    def test_is_deterministic_for_a_seed(self):
        assert subsample_to(range(100), 30, 42) == subsample_to(range(100), 30, 42)
