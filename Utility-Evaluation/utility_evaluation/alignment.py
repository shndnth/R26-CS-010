"""Label alignment: does each box sit on the object its class says it does."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utility_evaluation.config import BACKGROUND_CLASS, CLASS_NAMES

COORD_TOLERANCE = 0.001
MIN_BOX_PIXELS = 30
MAJORITY_THRESHOLD = 0.30

MASK_NAMES = {**CLASS_NAMES, BACKGROUND_CLASS: "background"}


@dataclass(frozen=True, slots=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_record(cls, record: dict) -> Box:
        return cls(
            int(record["class_id"]),
            float(record["x_center_norm"]),
            float(record["y_center_norm"]),
            float(record["width_norm"]),
            float(record["height_norm"]),
        )


def coordinates_valid(box: Box) -> bool:
    x1, x2 = box.x_center - box.width / 2, box.x_center + box.width / 2
    y1, y2 = box.y_center - box.height / 2, box.y_center + box.height / 2
    return (
        box.width > 0
        and box.height > 0
        and x1 >= -COORD_TOLERANCE
        and x2 <= 1 + COORD_TOLERANCE
        and y1 >= -COORD_TOLERANCE
        and y2 <= 1 + COORD_TOLERANCE
    )


def load_mask(path: Path) -> np.ndarray:
    """Class mask as a 2D array of class ids, whatever mode the PNG was saved in."""
    with Image.open(path) as image:
        array = np.array(image)
    return array[..., 0] if array.ndim == 3 else array


def box_region(box: Box, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    x1 = max(0, int((box.x_center - box.width / 2) * width))
    x2 = min(width, int((box.x_center + box.width / 2) * width))
    y1 = max(0, int((box.y_center - box.height / 2) * height))
    y2 = min(height, int((box.y_center + box.height / 2) * height))
    return mask[y1:y2, x1:x2]


def match_fraction(box: Box, region: np.ndarray) -> float:
    return float(np.count_nonzero(region == box.class_id)) / region.size


def spatial_status(box: Box, mask: np.ndarray) -> str:
    region = box_region(box, mask)
    if region.size < MIN_BOX_PIXELS:
        return "SKIPPED (region too small)"
    fraction = match_fraction(box, region)
    if fraction >= MAJORITY_THRESHOLD:
        return "ALIGNED"
    return f"MISALIGNED (only {fraction:.1%} of region matches class {box.class_id})"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    match_fraction: float
    dominant_class: int
    dominant_fraction: float
    kind: str
    category: str


def diagnose(box: Box, mask: np.ndarray) -> Diagnosis | None:
    """Explain a misaligned box."""
    if not coordinates_valid(box):
        return None
    region = box_region(box, mask)
    if region.size < MIN_BOX_PIXELS:
        return None
    fraction = match_fraction(box, region)
    if fraction >= MAJORITY_THRESHOLD:
        return None

    values, counts = np.unique(region, return_counts=True)
    dominant = int(values[np.argmax(counts)])
    dominant_fraction = float(counts.max()) / region.size

    if dominant == BACKGROUND_CLASS:
        kind, category = "occlusion_explained", "OCCLUSION-EXPLAINED (region dominated by background)"
    elif dominant in CLASS_NAMES and dominant != box.class_id:
        kind = "true_misalignment"
        category = (
            f"TRUE MISALIGNMENT (region dominated by {class_label(dominant)}, "
            f"not {class_label(box.class_id)})"
        )
    else:
        kind, category = "other", "OTHER / INCONCLUSIVE"
    return Diagnosis(fraction, dominant, dominant_fraction, kind, category)


def class_label(class_id: int) -> str:
    return MASK_NAMES.get(class_id, f"value {class_id}")
