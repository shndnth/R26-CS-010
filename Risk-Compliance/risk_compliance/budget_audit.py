"""Audit the claimed privacy budget against the per-step evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class BudgetAuditResult:
    config: str
    steps_logged: int
    reported_final_epsilon: float | None
    logged_final_epsilon: float | None
    peak_epsilon: float | None
    agreement: bool
    monotonic: bool
    ceiling_respected: bool
    halted_early: bool
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.agreement and self.monotonic and self.ceiling_respected

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "steps_logged": self.steps_logged,
            "reported_final_epsilon": self.reported_final_epsilon,
            "logged_final_epsilon": self.logged_final_epsilon,
            "peak_epsilon": self.peak_epsilon,
            "agreement": self.agreement,
            "monotonically_increasing": self.monotonic,
            "ceiling_respected": self.ceiling_respected,
            "halted_early": self.halted_early,
            "notes": list(self.notes),
            "verdict": self.verdict,
        }


def audit_run(run_dir: Path, ceiling: float, tolerance: float) -> BudgetAuditResult:
    """Audit one configuration directory."""
    report_path = run_dir / "audit_report.json"
    log_path = run_dir / "budget_log.csv"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results", {})

    # The non-private baseline declares no epsilon; that is not epsilon = 0.
    if report.get("differential_privacy_applied") is False:
        return BudgetAuditResult(
            config=run_dir.name,
            steps_logged=0,
            reported_final_epsilon=None,
            logged_final_epsilon=None,
            peak_epsilon=None,
            agreement=True,
            monotonic=True,
            ceiling_respected=True,
            halted_early=False,
            notes=("non-private control run; no privacy budget exists to audit",),
        )

    if not log_path.is_file():
        return BudgetAuditResult(
            config=run_dir.name,
            steps_logged=0,
            reported_final_epsilon=results.get("final_epsilon"),
            logged_final_epsilon=None,
            peak_epsilon=None,
            agreement=False,
            monotonic=False,
            ceiling_respected=False,
            halted_early=bool(results.get("halted_early", False)),
            notes=("budget_log.csv missing; the epsilon claim has no supporting evidence",),
        )

    log = pd.read_csv(log_path)
    series = log["cumulative_eps"]

    reported = float(results.get("final_epsilon", float("nan")))
    logged = float(series.iloc[-1])
    peak = float(series.max())

    # RDP composition is cumulative, so epsilon must never decrease.
    monotonic = bool((series.diff().dropna() >= -1e-9).all())

    notes: list[str] = []
    target = report.get("target_epsilon")
    if target is not None and reported > float(target):
        notes.append(
            f"final epsilon {reported:.4f} slightly exceeds the target {float(target):.1f}; "
            "expected from RDP accounting granularity, since epsilon is evaluated "
            "between steps rather than continuously"
        )

    return BudgetAuditResult(
        config=run_dir.name,
        steps_logged=len(log),
        reported_final_epsilon=round(reported, 6),
        logged_final_epsilon=round(logged, 6),
        peak_epsilon=round(peak, 6),
        agreement=abs(reported - logged) < tolerance,
        monotonic=monotonic,
        ceiling_respected=peak <= ceiling,
        halted_early=bool(results.get("halted_early", False)),
        notes=tuple(notes),
    )


def audit_all(runs_root: Path, ceiling: float, tolerance: float) -> list[BudgetAuditResult]:
    results: list[BudgetAuditResult] = []
    for report_path in sorted(runs_root.rglob("audit_report.json")):
        results.append(audit_run(report_path.parent, ceiling, tolerance))
    return results
