"""Check that the calibration queries compose to the epsilon the calibration stage claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_compliance.composition import analyse


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration privacy budget composition")
    parser.add_argument("--audit-log", type=Path, required=True, help="calibration_audit_log.json")
    parser.add_argument("--published", type=Path, default=None,
                        help="calibration_stats.json; its calibration_epsilon_spent is the claim")
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration_composition.json"))
    args = parser.parse_args()

    audit_log = json.loads(args.audit_log.read_text(encoding="utf-8"))
    published = json.loads(args.published.read_text(encoding="utf-8")) if args.published else None
    result = analyse(audit_log, published)

    print(f"epsilon_used holds: {result.field_meaning}\n")
    print(f"{'query':<30}{'sensitivity':>12}{'scale':>10}{'epsilon':>10}")
    for q in result.queries:
        print(f"{q.query:<30}{q.sensitivity:>12.3f}{q.laplace_scale:>10.4f}{q.epsilon:>10.4f}")
    print(f"\nclaimed total    {result.claimed_total:.4f}")
    print(f"composed total   {result.composed_total:.4f}  (sequential composition)")
    print(f"\nVERDICT: {result.verdict}")
    print(result.interpretation())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
