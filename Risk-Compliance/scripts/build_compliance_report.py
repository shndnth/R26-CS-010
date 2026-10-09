"""Assemble the final compliance report from the individual check results."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from risk_compliance.mapping import DECLARED_WEAKNESSES, MAPPINGS
from risk_compliance.settings import settings

CHECK_SOURCES: dict[str, tuple[str, str, str]] = {
    "certificate_verification": ("own", "certificate_verification.json", "all_valid"),
    "budget_audit": ("own", "budget_audit.json", "all_passed"),
    "calibration_audit_log": ("own", "reconstruction_attack.json", "audit_log_reconciliation"),
    "calibration_composition": ("own", "calibration_composition.json", "verdict"),
    "k_anonymity": ("own", "anonymity.json", "verdict"),
    "manifest_verification": ("own", "manifest_verification.json", "all_valid"),
    "poisoning_detection": ("utility", "poisoning_audit.json", "verdict"),
}

# FILTERED: the poisoning gate removed rejected frames, which is the control working.
PASSING_VERDICTS = frozenset({"PASS", "FILTERED"})


def load_check(directory: Path | None, filename: str) -> dict | None:
    if directory is None:
        return None
    path = directory / filename
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_passed(payload: dict | None, key: str) -> bool | None:
    """Interpret a check result. None means the check was not run."""
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.upper() in PASSING_VERDICTS
    if isinstance(value, dict):
        return bool(value.get("all_reconcile"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compliance report")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/compliance_report.json"))
    parser.add_argument("--utility-dir", type=Path, default=None,
                        help="Utility Evaluation outputs (poisoning_audit.json, utility_results.json)")
    args = parser.parse_args()

    cfg = settings()
    locations = {"own": args.outputs_dir, "utility": args.utility_dir}
    loaded = {
        name: load_check(locations[where], filename)
        for name, (where, filename, _) in CHECK_SOURCES.items()
    }
    verified = {
        name: check_passed(loaded[name], key)
        for name, (_, _, key) in CHECK_SOURCES.items()
    }
    utility = load_check(args.utility_dir, "utility_results.json")

    mapping_entries = [m.as_dict(verified.get(m.evidence_check)) for m in MAPPINGS]

    checks_run = [name for name, payload in loaded.items() if payload is not None]
    checks_missing = [name for name, payload in loaded.items() if payload is None]
    failures = [name for name, ok in verified.items() if ok is False]

    if failures:
        verdict = "FAIL"
    elif checks_missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    report = {
        "project": "R26-CS-010",
        "auditor": "IT22066916",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": (
            "Independent compliance verification of the privacy-enhanced synthetic AV "
            "data generation pipeline. Covers audit trail integrity, privacy budget "
            "accounting, calibration mechanism validation, dataset integrity, training "
            "data poisoning controls and k-anonymity of released metadata."
        ),
        "thresholds": {
            "epsilon_ceiling": cfg.thresholds.epsilon_ceiling,
            "mia_success_ceiling_pct": cfg.thresholds.mia_success_ceiling,
            "reid_ceiling_pct": cfg.thresholds.reid_ceiling,
        },
        "checks_run": checks_run,
        "checks_not_run": checks_missing,
        "check_results": {name: payload for name, payload in loaded.items() if payload},
        "utility_evidence": (
            {
                "utility_retention_pct": utility.get("utility_retention_pct"),
                "fid": utility.get("fid"),
                "experiments_missing": utility.get("experiments_missing"),
            }
            if utility else None
        ),
        "regulatory_mapping": mapping_entries,
        "declared_weaknesses": list(DECLARED_WEAKNESSES),
        "overall_verdict": verdict,
        "verdict_note": (
            "PASS requires every check to have been run and passed. INCOMPLETE means "
            "one or more checks are outstanding, which is not the same as a failure. "
            "This verdict covers the checks listed above; it is not a statement that "
            "the system satisfies every obligation under GDPR or the PDPA."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'provision':<22}{'requirement':<42}verified")
    print("-" * 76)
    for entry in mapping_entries:
        state = {True: "yes", False: "NO", None: "not run"}[entry["verified"]]
        print(f"{entry['provision']:<22}{entry['requirement'][:40]:<42}{state}")

    print(f"\nchecks run      {', '.join(checks_run) if checks_run else 'none'}")
    if checks_missing:
        print(f"checks not run  {', '.join(checks_missing)}")
    print(f"\ndeclared weaknesses  {len(DECLARED_WEAKNESSES)}")
    for weakness in DECLARED_WEAKNESSES:
        print(f"  [{weakness['severity']}] {weakness['issue']}")

    print(f"\nOVERALL: {verdict}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
