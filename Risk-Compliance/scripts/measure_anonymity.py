"""Measure k-anonymity of the released metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_compliance.anonymity import load_metadata, measure
from risk_compliance.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="k-anonymity measurement")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory containing *_metadata.csv files")
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/anonymity.json"))
    args = parser.parse_args()

    cfg = settings().anonymity
    threshold = args.threshold if args.threshold is not None else cfg.k_threshold

    data = load_metadata(args.data_dir)
    result = measure(data, cfg.quasi_identifiers, threshold)

    print(f"rows                    {result.total_rows}")
    print(f"quasi-identifiers       {', '.join(result.quasi_identifiers)}")
    print(f"distinct combinations   {result.distinct_combinations}")
    print(f"k, smallest class       {result.k}")
    print(f"unique combinations     {result.singletons}")
    print(f"classes below k={threshold:<10}{result.below_threshold}")
    print(f"\nVERDICT: {result.verdict}")
    print(f"\n{result.interpretation()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
