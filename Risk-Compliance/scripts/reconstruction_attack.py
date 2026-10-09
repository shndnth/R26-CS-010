"""Reconstruction attack on the calibration channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_compliance.reconstruction import run_attack, verify_audit_log
from risk_compliance.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration reconstruction attack")
    parser.add_argument("--true-values", type=Path, required=True,
                        help="unprivatised source statistics; never commit this file")
    parser.add_argument("--published", type=Path, required=True,
                        help="calibration_stats.json as delivered to the simulator")
    parser.add_argument("--audit-log", type=Path, default=None,
                        help="calibration_audit_log.json, enables the reconciliation check")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/reconstruction_attack.json"))
    args = parser.parse_args()

    cfg = settings().calibration
    epsilon = args.epsilon if args.epsilon is not None else cfg.epsilon

    result = run_attack(args.true_values, args.published, epsilon, cfg.sensitivities)

    print(f"epsilon {epsilon}\n")
    print(f"{'field':<30}{'true':>10}{'published':>12}{'error':>10}{'expected':>10}{'ratio':>8}")
    print("-" * 80)
    for f in result.fields:
        print(
            f"{f.field:<30}{f.true_value:>10.4f}{f.published_value:>12.4f}"
            f"{f.absolute_error:>10.4f}{f.expected_error:>10.4f}{f.ratio:>8.2f}"
        )
    print("-" * 80)
    print(f"{'mean':<30}{'':>10}{'':>12}"
          f"{result.mean_absolute_error:>10.4f}{result.mean_expected_error:>10.4f}"
          f"{result.calibration_ratio:>8.2f}")

    print(f"\n{result.interpretation()}")

    print("\nrelative distortion of each published value")
    for f in result.fields:
        print(f"  {f.field:<30}{f.relative_distortion_pct:>8.1f}%")

    payload = result.as_dict()

    if args.audit_log and args.audit_log.is_file():
        print("\naudit log reconciliation")
        reconciliation = verify_audit_log(args.true_values, args.published, args.audit_log)
        for check in reconciliation["checks"]:
            mark = "ok" if check["reconciles"] else "MISMATCH"
            print(f"  {check['field']:<30}{mark}")
        print(f"\n  {reconciliation['interpretation']}")
        payload["audit_log_reconciliation"] = reconciliation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
