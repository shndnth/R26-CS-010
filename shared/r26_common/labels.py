"""YOLO label files and the CARLA frame layout."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class ObjectClass(IntEnum):
    CAR = 0
    PEDESTRIAN = 1
    CYCLIST = 2


@dataclass(frozen=True, slots=True)
class BoundingBox:
    object_class: int
    x_center: float
    y_center: float
    width: float
    height: float

    def area_native(self, native_width: int, native_height: int) -> float:
        return (self.width * native_width) * (self.height * native_height)

    @property
    def is_within_frame(self) -> bool:
        return (
            self.x_center - self.width / 2 >= -0.01
            and self.x_center + self.width / 2 <= 1.01
            and self.y_center - self.height / 2 >= -0.01
            and self.y_center + self.height / 2 <= 1.01
        )


def parse_bbox_file(path: Path) -> list[BoundingBox]:
    """Parse a label file. Malformed lines are skipped; validation is the poisoning detector's job."""
    if not path.is_file():
        return []
    boxes: list[BoundingBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append(BoundingBox(int(parts[0]), *(float(v) for v in parts[1:5])))
        except ValueError:
            continue
    return boxes


def format_bbox_line(box: BoundingBox) -> str:
    return (
        f"{box.object_class} {box.x_center:.6f} {box.y_center:.6f} "
        f"{box.width:.6f} {box.height:.6f}"
    )


def iter_frame_pairs(town_dir: Path, labels_dir: Path | None = None) -> Iterator[tuple[Path, Path, str]]:
    """(rgb_path, label_path, frame_id) for every frame in a town folder, in frame order."""
    labels = labels_dir or town_dir / "labels"
    for rgb_path in sorted(town_dir.glob("rgb_f*.png")):
        frame_id = rgb_path.stem.removeprefix("rgb_f")
        yield rgb_path, labels / f"bbox_f{frame_id}.txt", frame_id
