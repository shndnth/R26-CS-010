"""KITTI to YOLO label conversion, for the real-data baseline."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from utility_evaluation.conversion import SplitCounts, prepare_target, split_pairs, write_data_yaml

# Tram, Misc and DontCare are intentionally dropped.
KITTI_CLASS_MAP: dict[str, int] = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}


def convert_label(label_path: Path, image_path: Path, out_path: Path) -> int:
    """Convert one KITTI label file. Returns the number of boxes written."""
    with Image.open(image_path) as image:
        width, height = image.size

    lines: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 8 or parts[0] not in KITTI_CLASS_MAP:
            continue

        class_id = KITTI_CLASS_MAP[parts[0]]
        try:
            left, top, right, bottom = (float(v) for v in parts[4:8])
        except ValueError:
            continue

        box_w = (right - left) / width
        box_h = (bottom - top) / height
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = ((left + right) / 2) / width
        y_center = ((top + bottom) / 2) / height
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def collect_kitti_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path, str]]:
    return [
        (image, labels_dir / f"{image.stem}.txt", image.stem)
        for image in sorted(images_dir.glob("*.png"))
        if (labels_dir / f"{image.stem}.txt").is_file()
    ]


def build_kitti_dataset(
    images_dir: Path,
    labels_dir: Path,
    target: Path,
    class_names: dict[int, str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
    overwrite: bool = False,
) -> SplitCounts:
    """Convert KITTI into the same layout and split proportions as the CARLA set."""
    pairs = collect_kitti_pairs(images_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(f"no image and label pairs found in {images_dir} and {labels_dir}")

    prepare_target(target, overwrite)
    splits = split_pairs(pairs, train_fraction, val_fraction, seed)
    for split, members in splits.items():
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image, label, uid in members:
            shutil.copy2(image, target / "images" / split / f"{uid}.png")
            convert_label(label, image, target / "labels" / split / f"{uid}.txt")

    write_data_yaml(target, class_names)
    return SplitCounts(*(len(splits[s]) for s in ("train", "val", "test")))
