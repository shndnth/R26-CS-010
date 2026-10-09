"""Convert KITTI into the same YOLO layout, classes and split proportions as the CARLA set."""

from __future__ import annotations

import argparse
from pathlib import Path

from utility_evaluation.config import CLASS_NAMES, SEED, SPLITS, add_root_argument, workspace_from
from utility_evaluation.kitti import build_kitti_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert KITTI to the YOLO layout")
    add_root_argument(parser)
    parser.add_argument("--images-dir", type=Path, help="default: <root>/kitti_raw/training/image_2")
    parser.add_argument("--labels-dir", type=Path, help="default: <root>/kitti_raw/training/label_2")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild kitti_dataset/")
    args = parser.parse_args()

    ws = workspace_from(args)
    images = args.images_dir or ws.kitti_raw / "image_2"
    labels = args.labels_dir or ws.kitti_raw / "label_2"

    try:
        counts = build_kitti_dataset(images, labels, ws.kitti_dataset, CLASS_NAMES,
                                     SPLITS["train"], SPLITS["val"], args.seed, args.overwrite)
    except (FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"total {counts.total} | train={counts.train} val={counts.val} test={counts.test}")
    print(f"written to {ws.kitti_dataset}")


if __name__ == "__main__":
    main()
