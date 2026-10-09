"""Step 6: train and evaluate one detector per privacy configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from utility_evaluation.study import SYNTH_TO_REAL, SYNTH_TO_SYNTH, PlanError, Study, load_plan


def _fmt(scores: dict | None) -> str:
    return "n/a" if not scores else f"{scores['map50_95']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-epsilon utility study")
    parser.add_argument("--plan", type=Path, required=True,
                        help="study plan, see epsilon_study.example.yaml")
    parser.add_argument("--only", nargs="+", help="variant names to run; the rest are kept")
    parser.add_argument("--force", action="store_true", help="decrypt, rebuild and retrain")
    args = parser.parse_args()

    try:
        study = Study(load_plan(args.plan))
        payload = study.run(only=args.only, force=args.force)
    except PlanError as exc:
        raise SystemExit(f"plan error: {exc}") from None

    print(f"\n{'variant':<12}{'epsilon':>9}{'synthetic':>12}{'real':>10}{'retention':>12}")
    for record in payload["variants"]:
        epsilon = "none" if record["epsilon"] is None else f"{record['epsilon']:g}"
        kept = record.get("utility_retention_pct")
        print(f"{record['name']:<12}{epsilon:>9}{_fmt(record[SYNTH_TO_SYNTH]):>12}"
              f"{_fmt(record[SYNTH_TO_REAL]):>10}{'n/a' if kept is None else f'{kept:.1f}%':>12}")
    if payload["real -> real"] is None:
        print("\nno KITTI baseline in the plan, so utility retention is not computed")
    print(f"\nwritten to {study.output}")


if __name__ == "__main__":
    main()
