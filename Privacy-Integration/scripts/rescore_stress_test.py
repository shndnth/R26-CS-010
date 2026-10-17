"""Recompute the stress-test verdict from a saved summary, without retraining.

Example:
    pi-stress-rescore
    pi-stress-rescore --summary outputs/stress_test/stress_test_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privacy_integration.attack.stress import SIGNIFICANCE_LEVEL, from_report
from privacy_integration.paths import outputs_root
from privacy_integration.serialization import read_json, utc_timestamp, write_json
from scripts.run_stress_test import print_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the stress-test verdict from saved results")
    parser.add_argument(
        "--summary", type=Path, default=None, help="Default: outputs/stress_test/stress_test_summary.json"
    )
    args = parser.parse_args()
    path = args.summary or outputs_root() / "stress_test" / "stress_test_summary.json"

    report = read_json(path)
    previous = report.get("verdict")
    summary = from_report(report)
    comparisons, verdict = summary.compare(), summary.verdict()

    if previous and "previous_verdicts" not in report:
        report["previous_verdicts"] = [{"verdict": previous, "generated_at": report.get("generated_at")}]
    report["comparisons"] = comparisons
    report["verdict"] = verdict
    report.setdefault("regime", {})["significance_level"] = SIGNIFICANCE_LEVEL
    report["rescored_at"] = utc_timestamp()
    write_json(report, path)

    print_results(comparisons, verdict)
    print(f"\nupdated {path} (the earlier verdict is kept under previous_verdicts)")


if __name__ == "__main__":
    main()
