"""Verify every signed audit report, then prove tampering is detected."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from r26_common.env import audit_hmac_key

from risk_compliance.verification import tamper_test, verify_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify signed audit certificates")
    parser.add_argument("--runs-dir", type=Path, required=True,
                        help="directory containing seed_NN/config_X folders")
    parser.add_argument("--tamper-target", type=Path, default=None,
                        help="run directory to use for the tamper test")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/certificate_verification.json"))
    parser.add_argument("--skip-tamper-test", action="store_true")
    args = parser.parse_args()

    key = audit_hmac_key()

    print("1. verifying signed audit reports")
    results = verify_all(args.runs_dir, key)
    if not results:
        raise SystemExit(f"no audit_report.json found under {args.runs_dir}")

    for result in results:
        print(f"   {result.report:<22} {result.verdict}")

    payload = {
        "reports_verified": len(results),
        "all_valid": all(r.passed for r in results),
        "results": [r.as_dict() for r in results],
    }

    if not args.skip_tamper_test:
        target = args.tamper_target
        if target is None:
            candidates = sorted(args.runs_dir.rglob("audit_report.json"))
            target = candidates[0].parent

        print("\n2. tamper test, simulating a falsified privacy claim")
        workspace = args.output.parent / "_tamper_test"
        original, result = tamper_test(
            target / "audit_report.json",
            target / "audit_certificate.json",
            key,
            workspace,
        )
        print(f"   recorded epsilon changed: {original} -> 0.01")
        print(f"   verification: {result.verdict}")
        if result.passed:
            print("   WARNING: tampering was NOT detected")
        else:
            print("   tampering detected and rejected")
        shutil.rmtree(workspace, ignore_errors=True)

        payload["tamper_test"] = {
            "target": target.relative_to(args.runs_dir).as_posix()
            if target.is_relative_to(args.runs_dir) else target.name,
            "original_value": original,
            "replacement_value": 0.01,
            "detected": not result.passed,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
