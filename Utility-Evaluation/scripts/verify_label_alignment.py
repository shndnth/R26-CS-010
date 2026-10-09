"""Step 5 (phase 1): baseline label alignment verification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utility_evaluation.alignment import Box, coordinates_valid, load_mask, spatial_status
from utility_evaluation.config import TOWNS, Workspace, add_root_argument, workspace_from

KEYS = ("frames", "total_boxes", "invalid_coords", "aligned", "misaligned",
        "skipped_small", "no_mask", "no_rgb")


def process_town(ws: Workspace, town: str) -> dict[str, int] | None:
    gt_path = ws.ground_truth / town / "ground_truth.json"
    if not gt_path.is_file():
        print(f"{town}: ground_truth.json not found (was Step 4 run?), skipping.")
        return None

    records = json.loads(gt_path.read_text(encoding="utf-8"))
    counts = dict.fromkeys(KEYS, 0)
    counts["frames"] = len(records)
    rows = []

    for record in records:
        rgb_ok = bool(record["rgb_path"]) and Path(record["rgb_path"]).is_file()
        counts["no_rgb"] += not rgb_ok
        mask_path = record["semantic_mask_path"]
        mask = load_mask(Path(mask_path)) if mask_path and Path(mask_path).is_file() else None
        counts["no_mask"] += mask is None

        for index, raw in enumerate(record["bounding_boxes"]):
            box = Box.from_record(raw)
            counts["total_boxes"] += 1
            valid = coordinates_valid(box)
            counts["invalid_coords"] += not valid
            status = spatial_status(box, mask) if mask is not None else "NO_MASK_AVAILABLE"
            if status == "ALIGNED":
                counts["aligned"] += 1
            elif status.startswith("MISALIGNED"):
                counts["misaligned"] += 1
            elif status.startswith("SKIPPED"):
                counts["skipped_small"] += 1
            rows.append([record["frame_id"], index, box.class_id, valid, rgb_ok, mask is not None, status])

    ws.alignment_report.mkdir(parents=True, exist_ok=True)
    report = ws.alignment_report / f"{town}_alignment_report.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "object_index", "class_id", "coordinates_valid",
                         "rgb_found", "mask_found", "alignment_status"])
        writer.writerows(rows)

    print(f"\n {town}: {counts['frames']} frames | {counts['total_boxes']} boxes checked")
    print(f"   Coordinate validity: {counts['total_boxes'] - counts['invalid_coords']}/"
          f"{counts['total_boxes']} valid ({counts['invalid_coords']} invalid)")
    print(f"   Spatial alignment:   {counts['aligned']} aligned | {counts['misaligned']} misaligned | "
          f"{counts['skipped_small']} skipped (too small) | frames missing mask: {counts['no_mask']} | "
          f"frames missing rgb: {counts['no_rgb']}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: label alignment verification")
    add_root_argument(parser)
    args = parser.parse_args()

    ws = workspace_from(args)
    print("Step 5 (Phase 1) - Baseline Label Alignment Verification")
    print(f"Source (Step 4 output): {ws.ground_truth}")
    print(f"Report output: {ws.alignment_report}")
    if not ws.ground_truth.is_dir():
        raise SystemExit(f"'{ws.ground_truth}' not found. Run Step 4 (extract_ground_truth.py) first.")

    totals = dict.fromkeys(KEYS, 0)
    lines = []
    for town in TOWNS:
        result = process_town(ws, town)
        if result:
            for key in KEYS:
                totals[key] += result[key]
            lines.append(f"{town}: {result}")

    checked = totals["aligned"] + totals["misaligned"]
    print("\n" + "=" * 55)
    print(f"DONE. Total boxes checked: {totals['total_boxes']}")
    print(f"   Invalid coordinates: {totals['invalid_coords']}")
    print(f"   Aligned: {totals['aligned']} | Misaligned: {totals['misaligned']} | "
          f"Skipped (too small): {totals['skipped_small']}")
    if checked:
        print(f"   Alignment rate among checkable boxes: {100 * totals['aligned'] / checked:.1f}%")
    print(f"   Frames missing mask: {totals['no_mask']} | Frames missing rgb: {totals['no_rgb']}")

    ws.alignment_report.mkdir(parents=True, exist_ok=True)
    (ws.alignment_report / "overall_summary.txt").write_text(
        "Step 5 (Phase 1) - Baseline Label Alignment Verification - Summary\n" + "=" * 55 + "\n"
        + "\n".join(lines) + f"\n\nTOTAL: {totals}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
