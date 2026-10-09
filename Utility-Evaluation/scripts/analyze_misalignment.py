"""Step 5 deep dive: why each misaligned box is misaligned."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utility_evaluation.alignment import Box, class_label, diagnose, load_mask
from utility_evaluation.config import TOWNS, Workspace, add_root_argument, workspace_from

KINDS = ("occlusion_explained", "true_misalignment", "other")


def process_town(ws: Workspace, town: str) -> dict[str, int] | None:
    gt_path = ws.ground_truth / town / "ground_truth.json"
    if not gt_path.is_file():
        print(f"{town}: ground_truth.json not found, skipping.")
        return None

    counts = dict.fromkeys(KINDS, 0)
    rows = []
    for record in json.loads(gt_path.read_text(encoding="utf-8")):
        mask_path = record["semantic_mask_path"]
        if not mask_path or not Path(mask_path).is_file():
            continue
        mask = load_mask(Path(mask_path))
        for index, raw in enumerate(record["bounding_boxes"]):
            box = Box.from_record(raw)
            diagnosis = diagnose(box, mask)
            if diagnosis is None:
                continue
            counts[diagnosis.kind] += 1
            rows.append([record["frame_id"], index, class_label(box.class_id),
                         f"{diagnosis.match_fraction:.1%}", class_label(diagnosis.dominant_class),
                         f"{diagnosis.dominant_fraction:.1%}", diagnosis.category])

    ws.alignment_report.mkdir(parents=True, exist_ok=True)
    report = ws.alignment_report / f"{town}_misalignment_breakdown.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "object_index", "labeled_class", "match_fraction_to_label",
                         "dominant_class_in_region", "dominant_fraction", "category"])
        writer.writerows(rows)

    total = sum(counts.values())
    counts["total"] = total
    print(f"\n{town}: {total} misaligned boxes analysed")
    if total:
        for key in KINDS:
            print(f"   {key:<20} {counts[key]} ({counts[key] / total:.1%})")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down misaligned boxes by cause")
    add_root_argument(parser)
    args = parser.parse_args()

    ws = workspace_from(args)
    print("Step 5 (Phase 1) - Misalignment Deep-Dive Analysis")
    print(f"Source: {ws.ground_truth}")
    if not ws.ground_truth.is_dir():
        raise SystemExit(f"'{ws.ground_truth}' not found. Run Step 4 first.")

    totals = dict.fromkeys((*KINDS, "total"), 0)
    lines = []
    for town in TOWNS:
        result = process_town(ws, town)
        if result:
            for key in totals:
                totals[key] += result[key]
            lines.append(f"{town}: {result}")

    print("\n" + "=" * 55)
    print(f"DONE. Total misaligned boxes analysed: {totals['total']}")
    if totals["total"]:
        for key in KINDS:
            print(f"   {key:<20} {totals[key]} ({totals[key] / totals['total']:.1%})")

    ws.alignment_report.mkdir(parents=True, exist_ok=True)
    (ws.alignment_report / "misalignment_breakdown_summary.txt").write_text(
        "Step 5 (Phase 1) - Misalignment Deep-Dive Analysis - Summary\n" + "=" * 55 + "\n"
        + "\n".join(lines) + f"\n\nTOTAL: {totals}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
