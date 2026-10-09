"""Step 3: data poisoning detection."""

from __future__ import annotations

import argparse
import csv
import json

from utility_evaluation.config import CLASS_NAMES, TOWNS, add_root_argument, workspace_from
from utility_evaluation.poisoning import Thresholds, audit, write_clean_labels


def main() -> None:
    defaults = Thresholds()
    parser = argparse.ArgumentParser(description="Step 3: data poisoning detection")
    add_root_argument(parser)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--alpha", type=float, default=defaults.chi2_alpha)
    parser.add_argument("--cramers-v", type=float, default=defaults.cramers_v)
    parser.add_argument("--audit-only", action="store_true",
                        help="report only, do not rewrite clean_labels/")
    args = parser.parse_args()

    ws = workspace_from(args)
    print("Step 3 - Data Poisoning Detection")
    print(f"Source:        {ws.decrypted}")
    print(f"Report output: {ws.poisoning_report}")
    if not ws.decrypted.is_dir():
        raise SystemExit(f"'{ws.decrypted}' not found. Run Step 2 (decrypt_all_images.py) first.")

    thresholds = Thresholds(batch_size=args.batch_size, chi2_alpha=args.alpha,
                            cramers_v=args.cramers_v)
    result = audit(ws.decrypted, TOWNS, thresholds)
    if not result.towns:
        raise SystemExit("no towns with a labels/ folder were found")

    ws.poisoning_report.mkdir(parents=True, exist_ok=True)
    for town, town_audit in result.towns.items():
        rates = town_audit.class_rates
        rejected = sum(1 for r, _ in town_audit.batches if r.rejected)
        print(f"\n {town} distribution [car, pedestrian, cyclist]: "
              f"[{rates[0]:.3f}, {rates[1]:.3f}, {rates[2]:.3f}] "
              f"({len(town_audit.batches)} batches, {sum(town_audit.class_counts.values())} objects)")
        print(f"   Batches: {rejected} rejected | {len(town_audit.batches) - rejected} accepted or skipped")
        print(f"   Frames failing structural checks: {len(town_audit.structurally_rejected)}")

        report = ws.poisoning_report / f"{town}_batch_report.csv"
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["batch_idx", "first_file", "last_file",
                             *(f"{n}_count" for n in CLASS_NAMES.values()),
                             "total_objects", "p_value", "cramers_v", "status"])
            for batch, _ in town_audit.batches:
                writer.writerow([
                    batch.index, batch.first_file, batch.last_file, *batch.counts, batch.total,
                    "N/A" if batch.p_value is None else round(batch.p_value, 6),
                    "N/A" if batch.cramers_v is None else round(batch.cramers_v, 4),
                    batch.status,
                ])

    (ws.poisoning_report / "structural_anomalies.txt").write_text(
        "\n".join(str(a) for a in result.anomalies) + "\n", encoding="utf-8"
    )

    if result.town_warnings:
        print("\nCross-town warnings (reported, not rejected; CARLA towns differ by design):")
        for warning in result.town_warnings:
            print(f"  {warning}")

    if not args.audit_only:
        written = write_clean_labels(result, ws.decrypted, ws.clean_labels)
        print(f"\nClean labels written: {written} files under {ws.clean_labels}")

    ws.results.mkdir(parents=True, exist_ok=True)
    (ws.results / "poisoning_audit.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print(f"   Frames scanned:          {result.frames_scanned}")
    print(f"   Boxes scanned:           {result.boxes_scanned}")
    print(f"   Structural anomalies:    {len(result.anomalies)}")
    print(f"   Batches rejected:        {result.batches_rejected}")
    print(f"   Frames excluded:         {result.frames_excluded}")
    print(f"   VERDICT: {result.verdict}")


if __name__ == "__main__":
    main()
