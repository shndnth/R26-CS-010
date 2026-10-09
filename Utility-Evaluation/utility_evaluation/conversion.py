"""Build the Ultralytics dataset layout from the decrypted CARLA frames."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from r26_common.labels import iter_frame_pairs

Pair = tuple[Path, Path, str]


@dataclass(frozen=True, slots=True)
class SplitCounts:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


def collect_pairs(data_dir: Path, towns: tuple[str, ...] | list[str],
                  clean_labels: Path | None = None) -> list[Pair]:
    """Gather (rgb, label, unique_id) for every usable frame."""
    pairs: list[Pair] = []
    for town in towns:
        town_dir = data_dir / town
        if not town_dir.is_dir():
            continue
        for rgb_path, raw_label, frame_id in iter_frame_pairs(town_dir):
            label = clean_labels / town / raw_label.name if clean_labels else raw_label
            if label.is_file():
                pairs.append((rgb_path, label, f"{town}_{frame_id}"))
    return pairs


def split_pairs(pairs: list[Pair], train_fraction: float, val_fraction: float, seed: int,
                mode: str = "random", block_size: int = 50) -> dict[str, list[Pair]]:
    """Assign pairs to train, val and test."""
    if mode == "random":
        units = [[pair] for pair in pairs]
    elif mode == "block":
        by_town: dict[str, list[Pair]] = {}
        for pair in pairs:
            by_town.setdefault(pair[2].split("_", 1)[0], []).append(pair)
        units = [
            frames[i : i + block_size]
            for _, frames in sorted(by_town.items())
            for i in range(0, len(frames), block_size)
        ]
    else:
        raise ValueError(f"unknown split mode '{mode}'")

    random.Random(seed).shuffle(units)
    n = len(pairs)
    targets = {"train": int(n * train_fraction), "val": int(n * val_fraction)}
    splits: dict[str, list[Pair]] = {"train": [], "val": [], "test": []}

    if mode == "random":
        flat = [unit[0] for unit in units]
        splits["train"] = flat[: targets["train"]]
        splits["val"] = flat[targets["train"] : targets["train"] + targets["val"]]
        splits["test"] = flat[targets["train"] + targets["val"] :]
        return splits

    for unit in units:
        if len(splits["train"]) < targets["train"]:
            splits["train"].extend(unit)
        elif len(splits["val"]) < targets["val"]:
            splits["val"].extend(unit)
        else:
            splits["test"].extend(unit)
    return splits


def prepare_target(target: Path, overwrite: bool) -> None:
    """Refuse to write into an existing dataset unless asked to replace it."""
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists. Pass --overwrite to delete and rebuild it."
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def write_data_yaml(target: Path, class_names: dict[int, str]) -> None:
    (target / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": class_names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def build_yolo_dataset(
    pairs: list[Pair],
    target: Path,
    class_names: dict[int, str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
    mode: str = "random",
    block_size: int = 50,
    overwrite: bool = False,
) -> SplitCounts:
    """Write images/, labels/ and data.yaml for every split."""
    prepare_target(target, overwrite)
    splits = split_pairs(pairs, train_fraction, val_fraction, seed, mode, block_size)

    for split, members in splits.items():
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rgb_path, label_path, uid in members:
            shutil.copy2(rgb_path, target / "images" / split / f"{uid}{rgb_path.suffix}")
            shutil.copy2(label_path, target / "labels" / split / f"{uid}.txt")

    write_data_yaml(target, class_names)
    return SplitCounts(*(len(splits[s]) for s in ("train", "val", "test")))
