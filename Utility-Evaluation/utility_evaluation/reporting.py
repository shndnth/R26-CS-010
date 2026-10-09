"""Aggregate evaluation results into the utility findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SYNTH_TO_SYNTH = "synthetic -> synthetic"
SYNTH_TO_REAL = "synthetic -> real"
REAL_TO_REAL = "real -> real"
REAL_TO_SYNTH = "real -> synthetic"

EXPERIMENT_ORDER = (SYNTH_TO_SYNTH, SYNTH_TO_REAL, REAL_TO_REAL, REAL_TO_SYNTH)


@dataclass(frozen=True, slots=True)
class Experiment:
    label: str
    map50: float
    map50_95: float
    per_class: dict[str, dict[str, float] | None]

    @classmethod
    def from_record(cls, record: dict) -> Experiment:
        return cls(
            label=record["label"],
            map50=float(record["map50"]),
            map50_95=float(record["map50_95"]),
            per_class=record.get("per_class", {}),
        )


@dataclass(frozen=True, slots=True)
class UtilityReport:
    experiments: dict[str, Experiment]
    fid: float | None = None
    fid_sample_size: int | None = None
    poisoning_verdict: str | None = None

    def _get(self, label: str) -> Experiment | None:
        return self.experiments.get(label)

    @property
    def retention_pct(self) -> float | None:
        """Synthetic-to-real performance as a percentage of the real-data baseline."""
        synthetic = self._get(SYNTH_TO_REAL)
        real = self._get(REAL_TO_REAL)
        if synthetic is None or real is None or real.map50_95 == 0:
            return None
        return 100.0 * synthetic.map50_95 / real.map50_95

    @property
    def sim_to_real_gap(self) -> float | None:
        """Absolute mAP50-95 drop when a synthetic-trained model meets real data."""
        synthetic = self._get(SYNTH_TO_SYNTH)
        transfer = self._get(SYNTH_TO_REAL)
        if synthetic is None or transfer is None:
            return None
        return synthetic.map50_95 - transfer.map50_95

    def weakest_class(self) -> tuple[str, float] | None:
        """The class the synthetic-trained model detects least well."""
        experiment = self._get(SYNTH_TO_SYNTH)
        if experiment is None:
            return None
        scored = [
            (name, values["ap50_95"])
            for name, values in experiment.per_class.items()
            if isinstance(values, dict) and "ap50_95" in values
        ]
        return min(scored, key=lambda pair: pair[1]) if scored else None

    def as_dict(self) -> dict:
        return {
            "experiments": {
                label: {
                    "map50": round(e.map50, 4),
                    "map50_95": round(e.map50_95, 4),
                    "per_class": e.per_class,
                }
                for label, e in self.experiments.items()
            },
            "utility_retention_pct": (
                round(self.retention_pct, 1) if self.retention_pct is not None else None
            ),
            "sim_to_real_gap_map50_95": (
                round(self.sim_to_real_gap, 4) if self.sim_to_real_gap is not None else None
            ),
            "weakest_class": self.weakest_class(),
            "fid": self.fid,
            "fid_sample_size": self.fid_sample_size,
            "poisoning_verdict": self.poisoning_verdict,
            "experiments_missing": [
                label for label in EXPERIMENT_ORDER if label not in self.experiments
            ],
        }

    def headline(self) -> str:
        retention = self.retention_pct
        if retention is None:
            return (
                "Utility retention cannot be computed yet. It needs both the "
                f"'{SYNTH_TO_REAL}' and '{REAL_TO_REAL}' experiments."
            )
        sentence = (
            f"A model trained purely on privacy-preserving synthetic data reaches "
            f"{retention:.0f}% of the detection performance of one trained on real data."
        )
        if retention > 100:
            sentence += (
                " Above 100% usually means the real-data baseline did not train properly;"
                " check its training curves before reporting this."
            )
        return sentence


def load(outputs_dir: Path) -> UtilityReport:
    """Assemble a report from whatever result files exist."""
    experiments: dict[str, Experiment] = {}

    evaluations_path = outputs_dir / "evaluations.json"
    if evaluations_path.is_file():
        for record in json.loads(evaluations_path.read_text(encoding="utf-8")):
            experiment = Experiment.from_record(record)
            experiments[experiment.label] = experiment

    fid = fid_n = None
    fid_path = outputs_dir / "fid.json"
    if fid_path.is_file():
        payload = json.loads(fid_path.read_text(encoding="utf-8"))
        fid = payload.get("fid")
        fid_n = min(payload.get("n_real", 0), payload.get("n_synthetic", 0)) or None

    poisoning = None
    poisoning_path = outputs_dir / "poisoning_audit.json"
    if poisoning_path.is_file():
        poisoning = json.loads(poisoning_path.read_text(encoding="utf-8")).get("verdict")

    return UtilityReport(
        experiments=experiments,
        fid=fid,
        fid_sample_size=fid_n,
        poisoning_verdict=poisoning,
    )
