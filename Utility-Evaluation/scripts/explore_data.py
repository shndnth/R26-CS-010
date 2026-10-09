"""Dataset statistics before trusting the data: frames, objects, class balance, object size."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from r26_common.labels import parse_bbox_file

from utility_evaluation.config import (
    CLASS_NAMES,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    TOWNS,
    add_root_argument,
    workspace_from,
)

SMALL_AREA = 32 * 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the decrypted CARLA labels")
    add_root_argument(parser)
    args = parser.parse_args()

    data = workspace_from(args).decrypted
    class_counts: Counter[int] = Counter()
    areas: dict[int, list[float]] = {c: [] for c in CLASS_NAMES}
    objects_per_frame: list[int] = []

    for town in TOWNS:
        for bbox_path in sorted((data / town / "labels").glob("bbox_*.txt")):
            boxes = parse_bbox_file(bbox_path)
            objects_per_frame.append(len(boxes))
            for box in boxes:
                class_counts[box.object_class] += 1
                areas.setdefault(box.object_class, []).append(
                    box.area_native(NATIVE_WIDTH, NATIVE_HEIGHT)
                )

    total = len(objects_per_frame)
    if not total:
        raise SystemExit(f"no label files found under {data}")
    empty = objects_per_frame.count(0)

    print(f"frames {total}")
    print(f"frames with 0 objects {empty} ({100 * empty / total:.1f}%)")
    print(f"mean objects/frame {np.mean(objects_per_frame):.2f}\n")
    for cid, count in sorted(class_counts.items()):
        values = np.array(areas[cid])
        print(f"{CLASS_NAMES.get(cid, cid)!s:<12} {count:>6} "
              f"median area {np.median(values):7.0f} px^2 "
              f"below 32x32: {100 * np.mean(values < SMALL_AREA):5.1f}%")


if __name__ == "__main__":
    main()
