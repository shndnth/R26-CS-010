"""Reconstruction attack on the calibration channel."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FieldReconstruction:
    field: str
    true_value: float
    published_value: float
    absolute_error: float
    expected_error: float
    relative_distortion_pct: float

    @property
    def ratio(self) -> float:
        """Observed error divided by theoretical expectation."""
        return self.absolute_error / self.expected_error if self.expected_error else float("nan")


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    fields: tuple[FieldReconstruction, ...]
    epsilon: float

    @property
    def mean_absolute_error(self) -> float:
        return sum(f.absolute_error for f in self.fields) / len(self.fields)

    @property
    def mean_expected_error(self) -> float:
        return sum(f.expected_error for f in self.fields) / len(self.fields)

    @property
    def calibration_ratio(self) -> float:
        """Near 1.0 means noise was injected at the magnitude theory requires."""
        expected = self.mean_expected_error
        return self.mean_absolute_error / expected if expected else float("nan")

    def interpretation(self) -> str:
        ratio = self.calibration_ratio
        if ratio < 0.3:
            return (
                "Observed noise is far below the theoretical expectation. This would "
                "indicate under-noising and a broken guarantee. Investigate before "
                "accepting the privacy claim."
            )
        if ratio > 3.0:
            return (
                "Observed noise substantially exceeds expectation. Not a privacy "
                "failure, but worth confirming the sensitivity values are correct."
            )
        return (
            "Observed noise is consistent with the magnitude the Laplace mechanism "
            "requires at this epsilon. Note this is a consistency check over a small "
            "number of queries, not a formal statistical test."
        )

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "n_fields": len(self.fields),
            "mean_absolute_error": round(self.mean_absolute_error, 4),
            "mean_expected_error": round(self.mean_expected_error, 4),
            "calibration_ratio": round(self.calibration_ratio, 4),
            "interpretation": self.interpretation(),
            "fields": [
                {
                    "field": f.field,
                    "true_value": f.true_value,
                    "published_value": f.published_value,
                    "absolute_error": round(f.absolute_error, 4),
                    "expected_error": round(f.expected_error, 4),
                    "ratio": round(f.ratio, 4),
                    "relative_distortion_pct": round(f.relative_distortion_pct, 2),
                }
                for f in self.fields
            ],
        }


def run_attack(
    true_path: Path,
    published_path: Path,
    epsilon: float,
    sensitivities: dict[str, float],
) -> ReconstructionResult:
    truth = json.loads(true_path.read_text(encoding="utf-8"))
    published = json.loads(published_path.read_text(encoding="utf-8"))

    fields: list[FieldReconstruction] = []
    for field, sensitivity in sensitivities.items():
        if field not in truth or field not in published:
            continue
        true_value = float(truth[field])
        published_value = float(published[field])
        error = abs(published_value - true_value)
        distortion = 100.0 * error / abs(true_value) if true_value else float("inf")

        fields.append(
            FieldReconstruction(
                field=field,
                true_value=true_value,
                published_value=published_value,
                absolute_error=error,
                expected_error=sensitivity / epsilon,
                relative_distortion_pct=distortion,
            )
        )

    if not fields:
        raise ValueError("no matching fields between the two calibration files")

    return ReconstructionResult(fields=tuple(fields), epsilon=epsilon)


def verify_audit_log(
    true_path: Path, published_path: Path, audit_log_path: Path
) -> dict:
    """Check that true value plus logged noise equals the published value."""
    truth = json.loads(true_path.read_text(encoding="utf-8"))
    published = json.loads(published_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_log_path.read_text(encoding="utf-8"))

    noise_by_field = {q["query"]: q.get("noise_added") for q in audit.get("queries", [])}

    checks = []
    all_reconcile = True
    for field, noise in noise_by_field.items():
        if field not in truth or field not in published or noise is None:
            continue
        if isinstance(noise, float) and math.isnan(noise):  # histogram queries carry no scalar noise
            continue
        expected = round(float(truth[field]) + float(noise), 4)
        actual = round(float(published[field]), 4)
        matches = abs(expected - actual) < 1e-4
        all_reconcile &= matches
        checks.append(
            {
                "field": field,
                "true_value": truth[field],
                "logged_noise": noise,
                "expected_published": expected,
                "actual_published": actual,
                "reconciles": matches,
            }
        )

    return {
        "all_reconcile": all_reconcile,
        "fields_checked": len(checks),
        "checks": checks,
        "interpretation": (
            "Every field satisfies true + logged_noise = published, so the audit log "
            "is an accurate record of the noise actually applied."
            if all_reconcile
            else "One or more fields do not reconcile. The audit log does not match "
                 "the published values and cannot be relied on."
        ),
    }
