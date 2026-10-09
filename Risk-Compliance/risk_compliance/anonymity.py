"""k-anonymity measurement on released metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class AnonymityResult:
    total_rows: int
    quasi_identifiers: tuple[str, ...]
    distinct_combinations: int
    k: int
    singletons: int
    below_threshold: int
    threshold: int

    @property
    def passed(self) -> bool:
        return self.k >= self.threshold

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def interpretation(self) -> str:
        if self.k == 1:
            return (
                f"{self.singletons} metadata combination(s) match exactly one frame, "
                "so those frames are uniquely identifiable from metadata alone."
            )
        if self.k < self.threshold:
            return (
                f"Smallest equivalence class is {self.k}, below the threshold of "
                f"{self.threshold}. Frames in small classes are more easily singled out."
            )
        return (
            f"Every metadata combination matches at least {self.k} frames, so no frame "
            "is distinguishable from its peers on these quasi-identifiers alone."
        )

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "total_rows": self.total_rows,
            "quasi_identifiers": list(self.quasi_identifiers),
            "distinct_combinations": self.distinct_combinations,
            "k": self.k,
            "singletons": self.singletons,
            "below_threshold": self.below_threshold,
            "threshold": self.threshold,
            "interpretation": self.interpretation(),
        }


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Concatenate every *_metadata.csv found under a directory."""
    frames = [pd.read_csv(p) for p in sorted(data_dir.rglob("*_metadata.csv"))]
    if not frames:
        raise FileNotFoundError(f"no *_metadata.csv found under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def measure(
    data: pd.DataFrame, quasi_identifiers: tuple[str, ...], threshold: int
) -> AnonymityResult:
    present = [column for column in quasi_identifiers if column in data.columns]
    missing = set(quasi_identifiers) - set(present)
    if missing:
        raise KeyError(f"quasi-identifier columns not present in the metadata: {missing}")

    group_sizes = data.groupby(list(present)).size()

    return AnonymityResult(
        total_rows=len(data),
        quasi_identifiers=tuple(present),
        distinct_combinations=len(group_sizes),
        k=int(group_sizes.min()),
        singletons=int((group_sizes == 1).sum()),
        below_threshold=int((group_sizes < threshold).sum()),
        threshold=threshold,
    )
