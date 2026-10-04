"""Size-filtered presence labels. Label file parsing lives in r26_common.labels."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from r26_common.labels import ObjectClass, parse_bbox_file

TARGET_CLASSES = frozenset({ObjectClass.PEDESTRIAN, ObjectClass.CYCLIST})


class Label(IntEnum):
    """Three-way frame label. EXCLUDED frames are dropped from the dataset."""

    NEGATIVE = 0
    POSITIVE = 1
    EXCLUDED = -1


def derive_label(
    bbox_path: Path,
    min_box_area_native: float,
    native_width: int,
    native_height: int,
) -> Label:
    """Derive a size-filtered presence label for one frame."""
    areas = [
        box.area_native(native_width, native_height)
        for box in parse_bbox_file(bbox_path)
        if box.object_class in TARGET_CLASSES
    ]
    if not areas:
        return Label.NEGATIVE
    return Label.POSITIVE if max(areas) >= min_box_area_native else Label.EXCLUDED
