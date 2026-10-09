"""Scan released artefacts for information that should not be published."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_compliance.leakage import scan
from risk_compliance.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata leakage scan")
    parser.add_argument("--root", type=Path, required=True,
                        help="directory of released artefacts to scan")
    parser.add_argument("--output", type=Path, default=Path("outputs/leakage_scan.json"))
    args = parser.parse_args()

    cfg = settings().leakage
    result = scan(args.root, cfg.scan_suffixes)

    print(f"scanned {result.files_scanned} files under {args.root}")
    print(f"findings {len(result.findings)}\n")

    for issue, items in sorted(result.by_issue().items()):
        print(f"{issue}: {len(items)}")
        for item in items[:3]:
            print(f"    {item.file}: {item.sample}")
        if len(items) > 3:
            print(f"    ... and {len(items) - 3} more")

    print(f"\nVERDICT: {result.verdict}")
    if result.findings:
        print("\nNot every finding is a breach. Absolute paths revealing usernames and")
        print("machine-specific details are worth reporting as findings with a")
        print("remediation, rather than treated as failures.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
