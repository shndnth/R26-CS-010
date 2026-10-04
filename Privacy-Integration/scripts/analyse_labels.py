"""Diagnose label quality by measuring target size at training resolution."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from r26_common.labels import ObjectClass, parse_bbox_file

from privacy_integration.data.labels import TARGET_CLASSES
from privacy_integration.logging_config import get_logger
from privacy_integration.settings import settings
from scripts._common import DEFAULT_TOWNS

logger = get_logger("labels")

SIZE_THRESHOLDS = (4, 16, 64, 144, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label quality diagnostic")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--towns", nargs="+", default=list(DEFAULT_TOWNS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = settings().dataset

    town_dirs = [args.data_dir / t for t in args.towns if (args.data_dir / t).is_dir()]
    if not town_dirs:
        raise SystemExit(f"no town folders under {args.data_dir}")

    sample = next(town_dirs[0].glob("rgb_f*.png"), None)
    if sample:
        with Image.open(sample) as image:
            logger.info(
                "native %dx%d -> training %dx%d (%.2fx, %.2fx)",
                *image.size, cfg.train_width, cfg.train_height,
                image.size[0] / cfg.train_width, image.size[1] / cfg.train_height,
            )

    class_counts: Counter[int] = Counter()
    largest_per_frame: list[float] = []
    frames = positives = 0
    scale = (cfg.train_width / cfg.native_width) * (cfg.train_height / cfg.native_height)

    for town_dir in town_dirs:
        labels_dir = town_dir / "labels"
        for rgb_path in sorted(town_dir.glob("rgb_f*.png")):
            frames += 1
            frame_id = rgb_path.stem.removeprefix("rgb_f")
            boxes = parse_bbox_file(labels_dir / f"bbox_f{frame_id}.txt")

            areas = []
            for box in boxes:
                class_counts[box.object_class] += 1
                if box.object_class in TARGET_CLASSES:
                    areas.append(box.area_native(cfg.native_width, cfg.native_height) * scale)
            if areas:
                positives += 1
                largest_per_frame.append(max(areas))
        logger.info("%s processed", town_dir.name)

    if not largest_per_frame:
        raise SystemExit("no pedestrian or cyclist boxes found")

    largest = np.asarray(largest_per_frame)
    print(f"\nframes={frames} with target={positives} ({100 * positives / frames:.1f}%)")
    for class_id, count in sorted(class_counts.items()):
        name = ObjectClass(class_id).name.lower() if class_id in set(ObjectClass) else str(class_id)
        print(f"  {name:<12}{count:>8}")

    print(f"\nlargest target per positive frame, px^2 at {cfg.train_width}x{cfg.train_height}:")
    for label, value in (
        ("min", largest.min()), ("p10", np.percentile(largest, 10)),
        ("median", np.median(largest)), ("p90", np.percentile(largest, 90)),
        ("max", largest.max()),
    ):
        print(f"  {label:<8}{value:10.1f}")

    print("\nfraction of positive frames below:")
    for threshold in SIZE_THRESHOLDS:
        side = int(threshold ** 0.5)
        print(f"  {threshold:>4} px^2 (~{side}x{side}): {100 * np.mean(largest < threshold):5.1f}%")

    tiny = 100 * float(np.mean(largest < 64))
    print()
    if tiny > 50:
        print(f"WARNING: {tiny:.1f}% of positive frames have targets below ~8x8 px.")
        print("Those labels are unlearnable at this resolution; accuracy is capped by")
        print("label quality. Raise resolution or the min box area filter.")
    elif tiny > 25:
        print(f"CAUTION: {tiny:.1f}% of positive frames have targets below ~8x8 px.")
    else:
        print(f"Label quality acceptable: only {tiny:.1f}% below ~8x8 px.")


if __name__ == "__main__":
    main()
