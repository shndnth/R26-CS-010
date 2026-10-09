"""Audit each claimed privacy budget against its per-step evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_compliance.budget_audit import audit_all
from risk_compliance.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent privacy budget audit")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/budget_audit.json"))
    args = parser.parse_args()

    cfg = settings().thresholds
    results = audit_all(args.runs_dir, cfg.epsilon_ceiling, cfg.epsilon_agreement_tolerance)
    if not results:
        raise SystemExit(f"no audit_report.json found under {args.runs_dir}")

    print(f"{'config':<12}{'steps':>8}{'reported':>12}{'logged':>12}{'peak':>10}  verdict")
    print("-" * 64)
    for r in results:
        reported = f"{r.reported_final_epsilon:.4f}" if r.reported_final_epsilon is not None else "n/a"
        logged = f"{r.logged_final_epsilon:.4f}" if r.logged_final_epsilon is not None else "n/a"
        peak = f"{r.peak_epsilon:.4f}" if r.peak_epsilon is not None else "n/a"
        print(f"{r.config:<12}{r.steps_logged:>8}{reported:>12}{logged:>12}{peak:>10}  {r.verdict}")

    notes = [(r.config, note) for r in results for note in r.notes]
    if notes:
        print("\nnotes")
        for config, note in notes:
            print(f"  {config}: {note}")

    print(f"\nceiling {cfg.epsilon_ceiling}")
    print("VERDICT:", "PASS" if all(r.passed for r in results) else "FAIL")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "epsilon_ceiling": cfg.epsilon_ceiling,
                "all_passed": all(r.passed for r in results),
                "results": [r.as_dict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
