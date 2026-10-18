"""Turn compliance_report.json into a signed, printable compliance certificate.

Examples:
    rc-certificate
    rc-certificate --report outputs/compliance_report.json --output outputs/compliance_certificate.pdf
    rc-certificate --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from risk_compliance.certificate import collect, render_pdf, sidecar, verify_certificate

HMAC_ENV = "R26_AUDIT_HMAC_KEY"


def _key() -> bytes | None:
    try:
        from r26_common.env import audit_hmac_key

        return audit_hmac_key()
    except Exception:  # noqa: BLE001 - no key means an unsigned certificate, stated on the page
        value = os.environ.get(HMAC_ENV)
        return value.encode("utf-8") if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the compliance certificate")
    parser.add_argument("--report", type=Path, default=Path("outputs/compliance_report.json"))
    parser.add_argument(
        "--output", type=Path, default=None, help="Default: compliance_certificate.pdf next to the report"
    )
    parser.add_argument("--verify", action="store_true", help="Check an existing certificate instead")
    args = parser.parse_args()

    pdf = args.output or args.report.with_name("compliance_certificate.pdf")
    record_path = pdf.with_suffix(".json")
    key = _key()

    if args.verify:
        result = verify_certificate(args.report, record_path, pdf, key)
        for name, value in result.as_dict().items():
            print(f"{name:<18}{'not checked' if value is None else value}")
        sys.exit(0 if result.passed else 1)

    if not args.report.is_file():
        raise SystemExit(f"no report at {args.report}; run rc-report first")
    data = collect(args.report, key)
    render_pdf(data, pdf)
    record = sidecar(data, pdf)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"certificate   {data.cert_id}")
    print(f"verdict       {data.verdict}")
    print(f"sha256        {data.digest}")
    print(f"signature     {data.signature or 'not signed (set ' + HMAC_ENV + ')'}")
    print(f"written to    {pdf}")
    print(f"record        {record_path}")


if __name__ == "__main__":
    main()
