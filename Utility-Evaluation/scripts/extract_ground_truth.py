"""Step 4: automated ground-truth extraction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from r26_common.labels import parse_bbox_file

from utility_evaluation.config import TOWNS, Workspace, add_root_argument, workspace_from


def frame_record(label_path: Path, town: str, town_dir: Path) -> dict:
    frame_id = label_path.stem.removeprefix("bbox_f")
    paths = {
        "semantic_mask_path": town_dir / "class_masks" / f"class_mask_f{frame_id}.png",
        "rgb_path": town_dir / f"rgb_f{frame_id}.png",
        "depth_path": town_dir / f"depth_f{frame_id}.png",
    }
    boxes = [
        {
            "class_id": b.object_class,
            "x_center_norm": b.x_center,
            "y_center_norm": b.y_center,
            "width_norm": b.width,
            "height_norm": b.height,
        }
        for b in parse_bbox_file(label_path)
    ]
    return {
        "frame_id": frame_id,
        "town": town,
        "num_objects": len(boxes),
        "bounding_boxes": boxes,
        **{key: (str(path) if path.is_file() else None) for key, path in paths.items()},
    }


def process_town(ws: Workspace, town: str) -> dict[str, int] | None:
    labels_dir = ws.clean_labels / town
    if not labels_dir.is_dir():
        print(f"{town}: no clean_labels found (was Step 3 run for this town?), skipping.")
        return None

    records = [
        frame_record(path, town, ws.decrypted / town)
        for path in sorted(labels_dir.glob("bbox_f*.txt"))
    ]

    out_dir = ws.ground_truth / town
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (out_dir / "extraction_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "num_objects", "semantic_mask_found", "rgb_found", "depth_found"])
        for r in records:
            writer.writerow([r["frame_id"], r["num_objects"], r["semantic_mask_path"] is not None,
                             r["rgb_path"] is not None, r["depth_path"] is not None])

    summary = {
        "frames": len(records),
        "objects": sum(r["num_objects"] for r in records),
        "missing_masks": sum(r["semantic_mask_path"] is None for r in records),
        "missing_rgb": sum(r["rgb_path"] is None for r in records),
    }
    print(f"\n{town}: {summary['frames']} frames | {summary['objects']} boxes | "
          f"{summary['missing_masks']} missing mask | {summary['missing_rgb']} missing rgb")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: ground-truth extraction")
    add_root_argument(parser)
    args = parser.parse_args()

    ws = workspace_from(args)
    print("Step 4 - Automated Ground-Truth Extraction Pipeline")
    print(f"Bounding boxes (Step 3 clean data): {ws.clean_labels}")
    print(f"Masks and images (Step 2 output):   {ws.decrypted}")
    print(f"Output: {ws.ground_truth}")
    if not ws.clean_labels.is_dir():
        raise SystemExit(f"'{ws.clean_labels}' not found. Run Step 3 (detect_poisoning.py) first.")

    totals = {"frames": 0, "objects": 0, "missing_masks": 0, "missing_rgb": 0}
    lines = []
    for town in TOWNS:
        result = process_town(ws, town)
        if result:
            for key in totals:
                totals[key] += result[key]
            lines.append(f"{town}: {result['frames']} frames, {result['objects']} objects, "
                         f"{result['missing_masks']} missing masks, {result['missing_rgb']} missing rgb")

    print("\n" + "=" * 55)
    print(f"   DONE. Total frames extracted: {totals['frames']}")
    print(f"   Total bounding boxes: {totals['objects']}")
    print(f"   Frames missing semantic mask: {totals['missing_masks']}")
    print(f"   Frames missing rgb image: {totals['missing_rgb']}")

    ws.ground_truth.mkdir(parents=True, exist_ok=True)
    (ws.ground_truth / "overall_summary.txt").write_text(
        "Step 4 - Automated Ground-Truth Extraction Pipeline - Summary\n" + "=" * 55 + "\n"
        + "\n".join(lines)
        + f"\n\nTOTAL: {totals['frames']} frames, {totals['objects']} bounding boxes, "
          f"{totals['missing_masks']} missing masks, {totals['missing_rgb']} missing rgb\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
