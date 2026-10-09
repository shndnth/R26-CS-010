"""Data poisoning detection, the security contribution of this component."""

from __future__ import annotations

import math
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utility_evaluation.config import CLASS_NAMES

VALID_CLASSES = frozenset(CLASS_NAMES)


@dataclass(frozen=True, slots=True)
class Thresholds:
    max_objects_per_frame: int = 50
    min_normalised_area: float = 1e-6
    batch_size: int = 100
    chi2_alpha: float = 0.0001
    cramers_v: float = 0.40
    min_objects_for_test: int = 10
    min_baseline_objects: int = 50
    town_z_limit: float = 4.0


@dataclass(frozen=True, slots=True)
class Anomaly:
    file: str
    line: int | None
    kind: str
    detail: str

    def __str__(self) -> str:
        location = f"{self.file}:{self.line}" if self.line else self.file
        return f"{location}  {self.kind}  {self.detail}"


@dataclass(frozen=True, slots=True)
class BatchResult:
    index: int
    first_file: str
    last_file: str
    counts: tuple[int, int, int]
    p_value: float | None
    cramers_v: float | None
    status: str

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def rejected(self) -> bool:
        return self.status == "REJECTED"


@dataclass
class TownAudit:
    town: str
    frames: list[str] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    structurally_rejected: set[str] = field(default_factory=set)
    batches: list[tuple[BatchResult, list[str]]] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def batch_rejected(self) -> set[str]:
        return {name for result, files in self.batches if result.rejected for name in files}

    @property
    def accepted(self) -> list[str]:
        excluded = self.structurally_rejected | self.batch_rejected
        return [name for name in self.frames if name not in excluded]

    @property
    def class_rates(self) -> dict[int, float]:
        total = sum(self.class_counts[c] for c in VALID_CLASSES)
        return {c: (self.class_counts[c] / total if total else 0.0) for c in sorted(VALID_CLASSES)}


@dataclass
class AuditResult:
    towns: dict[str, TownAudit] = field(default_factory=dict)
    town_warnings: list[Anomaly] = field(default_factory=list)

    @property
    def frames_scanned(self) -> int:
        return sum(len(t.frames) for t in self.towns.values())

    @property
    def boxes_scanned(self) -> int:
        return sum(sum(t.class_counts.values()) for t in self.towns.values())

    @property
    def anomalies(self) -> list[Anomaly]:
        return [a for t in self.towns.values() for a in t.anomalies]

    @property
    def frames_excluded(self) -> int:
        return self.frames_scanned - sum(len(t.accepted) for t in self.towns.values())

    @property
    def batches_rejected(self) -> int:
        return sum(1 for t in self.towns.values() for r, _ in t.batches if r.rejected)

    @property
    def verdict(self) -> str:
        """PASS when nothing was removed, FILTERED when a clean subset was produced."""
        return "PASS" if self.frames_excluded == 0 else "FILTERED"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "frames_scanned": self.frames_scanned,
            "boxes_scanned": self.boxes_scanned,
            "frames_excluded": self.frames_excluded,
            "structural_anomalies": len(self.anomalies),
            "batches_rejected": self.batches_rejected,
            "town_warnings": [str(w) for w in self.town_warnings],
            "anomalies": [str(a) for a in self.anomalies[:200]],
            "towns": {
                name: {
                    "frames": len(t.frames),
                    "accepted": len(t.accepted),
                    "structurally_rejected": len(t.structurally_rejected),
                    "batch_rejected_frames": len(t.batch_rejected),
                    "class_rates": {CLASS_NAMES[c]: round(v, 4) for c, v in t.class_rates.items()},
                }
                for name, t in self.towns.items()
            },
        }


def scan_frame(
    bbox_path: Path,
    max_objects: int = 50,
    min_area: float = 1e-6,
) -> tuple[list[int], list[Anomaly]]:
    """Structural validation of one label file. An empty file is valid."""
    anomalies: list[Anomaly] = []
    classes: list[int] = []
    name = bbox_path.name

    if not bbox_path.is_file():
        return classes, anomalies

    lines = [ln for ln in bbox_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if len(lines) > max_objects:
        anomalies.append(
            Anomaly(name, None, "implausible_count",
                    f"{len(lines)} objects exceeds the maximum of {max_objects}")
        )

    for number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            anomalies.append(Anomaly(name, number, "malformed", f"expected 5 fields, got {len(parts)}"))
            continue
        try:
            cid = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            anomalies.append(Anomaly(name, number, "non_numeric", line[:60]))
            continue
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            anomalies.append(Anomaly(name, number, "non_numeric", "nan or inf"))
            continue

        if cid not in VALID_CLASSES:
            anomalies.append(Anomaly(name, number, "unknown_class", f"class id {cid}"))
        for label, value in (("x", xc), ("y", yc), ("w", w), ("h", h)):
            if not 0.0 <= value <= 1.0:
                anomalies.append(Anomaly(name, number, "out_of_range", f"{label}={value}"))
        if w * h < min_area:
            anomalies.append(Anomaly(name, number, "degenerate_box", f"area {w * h:.2e}"))
        if not (xc - w / 2 >= -0.01 and xc + w / 2 <= 1.01):
            anomalies.append(Anomaly(name, number, "outside_frame", "horizontally"))
        if not (yc - h / 2 >= -0.01 and yc + h / 2 <= 1.01):
            anomalies.append(Anomaly(name, number, "outside_frame", "vertically"))

        classes.append(cid)

    return classes, anomalies


def chi_square_batch(
    counts: list[int],
    baseline: list[int],
    thresholds: Thresholds,
) -> tuple[float | None, float | None, str]:
    """Chi-square goodness of fit of one batch against its leave-one-out baseline."""
    from scipy.stats import chisquare

    total = sum(counts)
    baseline_total = sum(baseline)
    if total < thresholds.min_objects_for_test or baseline_total < thresholds.min_baseline_objects:
        return None, None, "SKIPPED"

    expected = [max(b / baseline_total * total, 1e-6) for b in baseline]
    scale = total / sum(expected)
    expected = [e * scale for e in expected]

    chi2, p_value = chisquare(f_obs=counts, f_exp=expected)
    cramers_v = math.sqrt(chi2 / (total * (len(counts) - 1)))

    rejected = p_value < thresholds.chi2_alpha and cramers_v > thresholds.cramers_v
    return float(p_value), float(cramers_v), "REJECTED" if rejected else "ACCEPTED"


def audit_town(labels_dir: Path, town: str, thresholds: Thresholds) -> TownAudit:
    result = TownAudit(town=town)
    result.frames = sorted(p.name for p in labels_dir.glob("bbox_f*.txt"))

    per_frame: dict[str, list[int]] = {}
    for name in result.frames:
        classes, anomalies = scan_frame(
            labels_dir / name, thresholds.max_objects_per_frame, thresholds.min_normalised_area
        )
        if anomalies:
            result.anomalies.extend(anomalies)
            result.structurally_rejected.add(name)
        valid = [c for c in classes if c in VALID_CLASSES]
        per_frame[name] = valid
        result.class_counts.update(valid)

    batches = [
        result.frames[i : i + thresholds.batch_size]
        for i in range(0, len(result.frames), thresholds.batch_size)
    ]
    batch_counts = []
    for files in batches:
        counts = [0, 0, 0]
        for name in files:
            for cid in per_frame[name]:
                counts[cid] += 1
        batch_counts.append(counts)

    town_totals = [sum(c[i] for c in batch_counts) for i in range(3)]
    for index, (files, counts) in enumerate(zip(batches, batch_counts, strict=True)):
        baseline = [town_totals[i] - counts[i] for i in range(3)]
        p_value, cramers_v, status = chi_square_batch(counts, baseline, thresholds)
        result.batches.append(
            (BatchResult(index, files[0], files[-1], tuple(counts), p_value, cramers_v, status), files)
        )

    return result


def town_outliers(rates_by_town: dict[str, dict[int, float]], z_limit: float) -> list[Anomaly]:
    """Flag a town whose class share sits far from the other towns."""
    warnings: list[Anomaly] = []
    towns = list(rates_by_town)
    if len(towns) < 3:
        return warnings

    for cid in sorted(VALID_CLASSES):
        for town in towns:
            value = rates_by_town[town][cid]
            others = np.array([rates_by_town[t][cid] for t in towns if t != town])
            mean, sd = float(others.mean()), float(others.std(ddof=1))
            if sd == 0:
                z = 0.0 if math.isclose(value, mean) else math.inf
            else:
                z = abs(value - mean) / sd
            if z > z_limit:
                warnings.append(
                    Anomaly(town, None, "town_outlier",
                            f"{CLASS_NAMES[cid]} share {value:.3f} vs {mean:.3f} elsewhere "
                            f"({z:.1f} sd)")
                )
    return warnings


def audit(labels_root: Path, towns: tuple[str, ...], thresholds: Thresholds | None = None) -> AuditResult:
    """Audit the ``labels/`` folder of every town under ``labels_root``."""
    thresholds = thresholds or Thresholds()
    result = AuditResult()
    for town in towns:
        labels_dir = labels_root / town / "labels"
        if labels_dir.is_dir():
            result.towns[town] = audit_town(labels_dir, town, thresholds)
    result.town_warnings = town_outliers(
        {name: t.class_rates for name, t in result.towns.items()}, thresholds.town_z_limit
    )
    return result


def write_clean_labels(result: AuditResult, labels_root: Path, clean_root: Path) -> int:
    """Copy accepted label files to ``clean_root/<town>``, rebuilt from scratch."""
    if clean_root.exists():
        shutil.rmtree(clean_root)
    written = 0
    for town, town_audit in result.towns.items():
        destination = clean_root / town
        destination.mkdir(parents=True, exist_ok=True)
        for name in town_audit.accepted:
            shutil.copy2(labels_root / town / "labels" / name, destination / name)
            written += 1
    return written
