"""Run-scoped output paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from privacy_integration.paths import ensure_dir, outputs_root

BASELINE_TAG = "no_dp"


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem layout for a single (seed, config) experiment."""

    root: Path
    seed: int
    tag: str

    @property
    def run_dir(self) -> Path:
        return self.root / f"seed_{self.seed}" / self.tag

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "model.pt"

    @property
    def metrics(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def budget_log(self) -> Path:
        return self.run_dir / "budget_log.csv"

    @property
    def audit_report(self) -> Path:
        return self.run_dir / "audit_report.json"

    @property
    def audit_certificate(self) -> Path:
        return self.run_dir / "audit_certificate.json"

    @property
    def mia_results(self) -> Path:
        return self.run_dir / "mia_results.json"

    def create(self) -> RunPaths:
        ensure_dir(self.run_dir)
        return self


class ArtifactStore:
    """Resolves run directories and discovers completed runs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or outputs_root() / "runs"

    def run(self, seed: int, tag: str) -> RunPaths:
        return RunPaths(root=self.root, seed=seed, tag=tag)

    def curve_path(self, seed: int) -> Path:
        return ensure_dir(self.root / f"seed_{seed}") / "calibration_curve.json"

    def summary_path(self) -> Path:
        return ensure_dir(self.root) / "summary.json"

    def figures_dir(self) -> Path:
        return ensure_dir(self.root.parent / "figures")

    def completed_seeds(self) -> list[int]:
        if not self.root.is_dir():
            return []
        seeds = []
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name.startswith("seed_"):
                try:
                    seeds.append(int(entry.name.removeprefix("seed_")))
                except ValueError:
                    continue
        return sorted(seeds)

    def completed_runs(self, seed: int) -> list[str]:
        seed_dir = self.root / f"seed_{seed}"
        if not seed_dir.is_dir():
            return []
        return sorted(d.name for d in seed_dir.iterdir() if d.is_dir() and (d / "metrics.json").is_file())
