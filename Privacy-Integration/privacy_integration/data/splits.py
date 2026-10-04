"""Stratified splitting and class rebalancing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DataSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    def __post_init__(self) -> None:
        train_s, val_s, test_s = set(self.train), set(self.val), set(self.test)
        overlap = (train_s & val_s) | (train_s & test_s) | (val_s & test_s)
        if overlap:
            raise ValueError(f"splits overlap on {len(overlap)} indices")

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.val), len(self.test)


def stratified_split(
    labels: Sequence[int], train_fraction: float, val_fraction: float, seed: int
) -> DataSplit:
    """Split indices while preserving class proportions in each partition."""
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("fractions must lie in (0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must leave a test split")

    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for class_value in sorted(set(labels)):
        indices = np.flatnonzero(label_array == class_value)
        rng.shuffle(indices)
        n_train = int(len(indices) * train_fraction)
        n_val = int(len(indices) * val_fraction)
        train.extend(indices[:n_train].tolist())
        val.extend(indices[n_train : n_train + n_val].tolist())
        test.extend(indices[n_train + n_val :].tolist())

    for part in (train, val, test):
        rng.shuffle(part)
    return DataSplit(train=train, val=val, test=test)


def balance_by_oversampling(indices: Sequence[int], labels: Sequence[int]) -> list[int]:
    """Balance a split by duplicating minority-class indices."""
    by_class: dict[int, list[int]] = {}
    for idx in indices:
        by_class.setdefault(labels[idx], []).append(idx)

    if len(by_class) < 2:
        return list(indices)

    largest = max(len(v) for v in by_class.values())
    balanced: list[int] = []
    for members in by_class.values():
        repeats, remainder = divmod(largest, len(members))
        balanced.extend(members * repeats)
        balanced.extend(members[:remainder])
    return balanced


def subsample_to(indices: Sequence[int], size: int, seed: int) -> list[int]:
    """Draw `size` indices without replacement, for balanced MIA evaluation."""
    if size >= len(indices):
        return list(indices)
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(indices), size=size, replace=False).tolist()
