"""CARLA frame dataset with size-filtered presence labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from privacy_integration.data.labels import Label, derive_label
from privacy_integration.logging_config import get_logger
from privacy_integration.settings import DatasetSettings

logger = get_logger(__name__)

# ImageNet statistics.
_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)


def build_transform(cfg: DatasetSettings) -> transforms.Compose:
    """Resize preserving the native aspect ratio, then normalise."""
    return transforms.Compose(
        [
            transforms.Resize((cfg.train_height, cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ]
    )


@dataclass(frozen=True, slots=True)
class DatasetStats:
    positive: int
    negative: int
    excluded: int
    gated: int = 0

    @property
    def usable(self) -> int:
        return self.positive + self.negative

    @property
    def scanned(self) -> int:
        return self.usable + self.excluded

    @property
    def positive_rate(self) -> float:
        return self.positive / self.usable if self.usable else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "excluded": self.excluded,
            "gated": self.gated,
            "usable": self.usable,
            "scanned": self.scanned,
            "positive_rate": round(self.positive_rate, 4),
        }


class CarlaFrameDataset(Dataset):
    """RGB frames paired with a binary pedestrian/cyclist presence label."""

    def __init__(
        self,
        town_dirs: Iterable[Path],
        cfg: DatasetSettings,
        transform: transforms.Compose | None = None,
        labels_root: Path | None = None,
    ) -> None:
        """With ``labels_root``, labels come from the poisoning gate's clean labels."""
        self._cfg = cfg
        self._transform = transform or build_transform(cfg)
        self._samples: list[tuple[Path, int]] = []

        positive = negative = excluded = gated = 0

        for town_dir in town_dirs:
            labels_dir = labels_root / town_dir.name if labels_root else town_dir / "labels"
            frames = sorted(town_dir.glob("rgb_f*.png"))
            if not frames:
                logger.warning("%s contains no rgb_f*.png frames", town_dir)
                continue

            for rgb_path in frames:
                frame_id = rgb_path.stem.removeprefix("rgb_f")
                label_path = labels_dir / f"bbox_f{frame_id}.txt"
                if labels_root is not None and not label_path.is_file():
                    gated += 1
                    continue
                label = derive_label(
                    label_path,
                    cfg.min_box_area_native,
                    cfg.native_width,
                    cfg.native_height,
                )
                if label is Label.EXCLUDED:
                    excluded += 1
                    continue
                if label is Label.POSITIVE:
                    positive += 1
                else:
                    negative += 1
                self._samples.append((rgb_path, int(label)))

            logger.info("%s | scanned %d frames", town_dir.name, len(frames))

        self.stats = DatasetStats(
            positive=positive, negative=negative, excluded=excluded, gated=gated
        )
        if gated:
            logger.info("%d frames skipped: rejected by the poisoning gate", gated)

        if not self._samples:
            raise ValueError(
                "dataset is empty after filtering; lower min_box_area_native or check --data-dir"
            )

        logger.info(
            "Dataset ready | usable=%d (positive=%d [%.1f%%]) excluded=%d [%.1f%% of %d] | "
            "%dx%d, min box %.0f px^2 native",
            self.stats.usable,
            self.stats.positive,
            100 * self.stats.positive_rate,
            self.stats.excluded,
            100 * self.stats.excluded / self.stats.scanned,
            self.stats.scanned,
            cfg.train_width,
            cfg.train_height,
            cfg.min_box_area_native,
        )

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self._samples]

    @property
    def paths(self) -> list[Path]:
        return [path for path, _ in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path, label = self._samples[index]
        with Image.open(rgb_path) as image:
            tensor = self._transform(image.convert("RGB"))
        return tensor, torch.tensor([float(label)])


def discover_town_dirs(data_dir: Path, towns: Iterable[str]) -> list[Path]:
    """Resolve town folder names to existing directories."""
    resolved = [data_dir / town for town in towns]
    missing = [d.name for d in resolved if not d.is_dir()]
    if missing:
        logger.warning("Skipping missing town folders: %s", ", ".join(missing))
    return [d for d in resolved if d.is_dir()]
