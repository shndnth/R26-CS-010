"""YOLOv8 training with one set of settings for every model in the study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utility_evaluation.config import SEED, pick_device


@dataclass(frozen=True, slots=True)
class TrainSettings:
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    patience: int = 15
    seed: int = SEED
    workers: int = 8
    device: str | None = None


def train(data_yaml: Path, runs_dir: Path, name: str, settings: TrainSettings) -> Path:
    """Train one model and return the path to its best weights."""
    from ultralytics import YOLO

    model = YOLO(settings.model)
    model.train(
        data=str(data_yaml),
        epochs=settings.epochs,
        imgsz=settings.imgsz,
        batch=settings.batch,
        seed=settings.seed,
        deterministic=True,
        project=str(runs_dir.resolve()),
        name=name,
        exist_ok=False,
        patience=settings.patience,
        device=pick_device(settings.device),
        workers=settings.workers,
    )
    return Path(model.trainer.best)
