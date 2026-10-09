"""Step 9: the joint privacy-utility figure and table."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from utility_evaluation.joint import build_rows, detection_measure, plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint privacy-utility figure")
    parser.add_argument("--study", type=Path, required=True, help="epsilon_study.json")
    parser.add_argument("--privacy-summary", type=Path, required=True,
                        help="summary.json from Privacy Integration (pi-report)")
    parser.add_argument("--output-dir", type=Path, help="default: the study's folder")
    args = parser.parse_args()

    study = json.loads(args.study.read_text(encoding="utf-8"))
    summary = json.loads(args.privacy_summary.read_text(encoding="utf-8"))
    rows = build_rows(study, summary)
    if not rows:
        raise SystemExit("nothing to plot: both inputs are empty")

    out = args.output_dir or args.study.parent
    out.mkdir(parents=True, exist_ok=True)
    key, _, title = detection_measure(rows)
    (out / "joint_privacy_utility.json").write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "detection_measure": key,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    plot(rows, out / "joint_privacy_utility.png")

    print(f"{'':<10}{'attack AUC':>12}{'classifier AUC':>16}{title:>32}")
    for r in rows:
        def show(value, pct=False):
            return "n/a" if value is None else (f"{value:.1f}%" if pct else f"{value:.4f}")
        print(f"{r['label']:<10}{show(r['attack_auc_mean']):>12}{show(r['classifier_auc_mean']):>16}"
              f"{show(r[key], key == 'utility_retention_pct'):>32}")
    print(f"\nwritten to {out / 'joint_privacy_utility.png'} and joint_privacy_utility.json")


if __name__ == "__main__":
    main()
