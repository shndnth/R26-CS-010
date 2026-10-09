"""Detection metrics: per-class results from Ultralytics and size-stratified AP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utility_evaluation.config import CLASS_NAMES

SIZE_BUCKETS: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}


def summarise_validation(metrics, label: str, **extra) -> dict:
    """Turn an Ultralytics validation result into a plain record."""
    box = metrics.box
    class_index = [int(c) for c in getattr(box, "ap_class_index", range(len(box.ap50)))]
    per_class: dict[str, dict[str, float] | None] = {name: None for name in CLASS_NAMES.values()}
    for position, class_id in enumerate(class_index):
        name = CLASS_NAMES.get(class_id)
        if name is not None:
            per_class[name] = {
                "ap50": round(float(box.ap50[position]), 4),
                "ap50_95": round(float(box.ap[position]), 4),
            }
    return {
        "label": label,
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "per_class": per_class,
        **extra,
    }


def bucket_of(area: float) -> str:
    for name, (low, high) in SIZE_BUCKETS.items():
        if low <= area < high:
            return name
    return "large"


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(tp_flags: list[int], num_gt: int) -> float | None:
    """All-points interpolated AP from TP/FP flags sorted by descending confidence."""
    if num_gt == 0:
        return None
    tp = np.asarray(tp_flags, dtype=float)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


Box = tuple[int, float, float, float, float]
Prediction = tuple[int, float, float, float, float, float]


def load_gt_boxes(label_path: Path, image_width: int, image_height: int) -> list[Box]:
    """(class_id, x1, y1, x2, y2) in pixels from a YOLO label file."""
    boxes: list[Box] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:])
        boxes.append((
            cid,
            (xc - w / 2) * image_width,
            (yc - h / 2) * image_height,
            (xc + w / 2) * image_width,
            (yc + h / 2) * image_height,
        ))
    return boxes


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


@dataclass
class SizeStratifiedAP:
    """AP50 per class and object size, following the COCO area-range rule."""

    iou_threshold: float = 0.5
    detections: dict[str, dict[int, list[tuple[float, int]]]] = field(init=False)
    gt_counts: dict[str, dict[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.detections = {b: {c: [] for c in CLASS_NAMES} for b in SIZE_BUCKETS}
        self.gt_counts = {b: {c: 0 for c in CLASS_NAMES} for b in SIZE_BUCKETS}

    def add_image(self, gt_boxes: list[Box], predictions: list[Prediction]) -> None:
        """Add one image. Predictions are (class_id, confidence, x1, y1, x2, y2)."""
        for bucket, (low, high) in SIZE_BUCKETS.items():
            for cid in CLASS_NAMES:
                gts = [g[1:] for g in gt_boxes if g[0] == cid]
                ignored = [not (low <= _area(g) < high) for g in gts]
                self.gt_counts[bucket][cid] += ignored.count(False)

                preds = sorted((p for p in predictions if p[0] == cid), key=lambda p: -p[1])
                matched = [False] * len(gts)
                for _, confidence, *coords in preds:
                    box = tuple(coords)
                    best, best_ignored = self._best_match(box, gts, matched, ignored)
                    if best is not None and not best_ignored:
                        matched[best] = True
                        self.detections[bucket][cid].append((confidence, 1))
                    elif best is not None:
                        matched[best] = True
                    elif low <= _area(box) < high:
                        self.detections[bucket][cid].append((confidence, 0))

    def _best_match(self, box, gts, matched, ignored) -> tuple[int | None, bool]:
        """Prefer an unmatched in-range ground truth; fall back to an ignored one."""
        for want_ignored in (False, True):
            best, best_iou = None, self.iou_threshold
            for j, gt in enumerate(gts):
                if matched[j] or ignored[j] != want_ignored:
                    continue
                overlap = iou(box, gt)
                if overlap >= best_iou:
                    best, best_iou = j, overlap
            if best is not None:
                return best, want_ignored
        return None, False

    def results(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for bucket in SIZE_BUCKETS:
            per_class = {}
            for cid, name in CLASS_NAMES.items():
                dets = sorted(self.detections[bucket][cid], key=lambda d: -d[0])
                ap = average_precision([d[1] for d in dets], self.gt_counts[bucket][cid])
                per_class[name] = {
                    "ap50": None if ap is None else round(ap, 4),
                    "gt_instances": self.gt_counts[bucket][cid],
                    "detections": len(dets),
                }
            scored = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
            out[bucket] = {
                "per_class": per_class,
                "map50": round(float(np.mean(scored)), 4) if scored else None,
            }
        return out


def evaluate(weights: Path, data_yaml: Path, label: str, device: str | None,
             project: Path, name: str, /, **extra) -> dict:
    """Test-split metrics for one model on one dataset, as a plain record."""
    from ultralytics import YOLO

    from utility_evaluation.config import pick_device

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", device=pick_device(device),
        project=str(project.resolve()), name=name,
    )
    return summarise_validation(metrics, label, **extra)
