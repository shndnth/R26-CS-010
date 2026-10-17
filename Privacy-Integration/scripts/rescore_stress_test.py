"""Recompute the stress-test verdict from a saved summary, without retraining.

Examples:
    pi-stress-rescore
    pi-stress-rescore --from-runs     # combine every seed run so far (e.g. 42-44 and 45-46)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privacy_integration.attack.stress import SIGNIFICANCE_LEVEL, from_report, from_run_files
from privacy_integration.paths import outputs_root
from privacy_integration.serialization import read_json, utc_timestamp, write_json
from scripts.run_stress_test import print_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the stress-test verdict from saved results")
    parser.add_argument(
        "--summary", type=Path, default=None, help="Default: outputs/stress_test/stress_test_summary.json"
    )
    parser.add_argument(
        "--from-runs",
        action="store_true",
        help="Rebuild from every per-run file (seed_<n>_<config>.json) next to the summary, "
        "so seeds run at different times are combined.",
    )
    args = parser.parse_args()
    path = args.summary or outputs_root() / "stress_test" / "stress_test_summary.json"

    report = read_json(path) if path.is_file() else {"regime": {}}
    previous = report.get("verdict")
    if args.from_runs:
        summary = from_run_files(path.parent)
        report["runs"] = [r.as_dict() for r in summary.runs]
        report.setdefault("regime", {})["seeds"] = summary.seeds()
        report["combined_from_run_files"] = True
    else:
        summary = from_report(report)
    comparisons, verdict, ordering = summary.compare(), summary.verdict(), summary.ordering()

    if previous and "previous_verdicts" not in report:
        report["previous_verdicts"] = [{"verdict": previous, "generated_at": report.get("generated_at")}]
    report["comparisons"] = comparisons
    report["verdict"] = verdict
    report["epsilon_ordering"] = ordering
    report.setdefault("regime", {})["significance_level"] = SIGNIFICANCE_LEVEL
    report["rescored_at"] = utc_timestamp()
    write_json(report, path)

    print_results(comparisons, verdict, ordering, summary.seeds())
    print(f"\nupdated {path} (the earlier verdict is kept under previous_verdicts)")


if __name__ == "__main__":
    main()
