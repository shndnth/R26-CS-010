"""Re-hash the decrypted dataset against the generator's SHA-256 manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_compliance.manifest import verify_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent manifest verification")
    parser.add_argument("--encrypted-dir", type=Path, required=True,
                        help="the delivered dataset, holding townXX/townXX_manifest.json")
    parser.add_argument("--decrypted-dir", type=Path, required=True,
                        help="the decrypted dataset to check")
    parser.add_argument("--towns", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/manifest_verification.json"))
    args = parser.parse_args()

    results = verify_dataset(args.encrypted_dir, args.decrypted_dir, args.towns)
    if not results:
        raise SystemExit(f"no town folders found under {args.encrypted_dir}")

    print(f"{'town':<10}{'entries':>9}{'verified':>10}{'mismatch':>10}{'missing':>9}  result")
    for r in results:
        state = "PASS" if r.passed else ("NO MANIFEST" if not r.manifest_found else "FAIL")
        counts = f"{r.entries:>9}{r.verified:>10}{len(r.mismatched):>10}{len(r.missing):>9}"
        print(f"{r.town:<10}{counts}  {state}")

    payload = {
        "towns_checked": len(results),
        "files_verified": sum(r.verified for r in results),
        "all_valid": all(r.passed for r in results),
        "towns": [r.as_dict() for r in results],
    }
    print(f"\nVERDICT: {'PASS' if payload['all_valid'] else 'FAIL'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
