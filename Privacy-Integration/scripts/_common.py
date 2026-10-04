"""Shared CLI plumbing for the runner scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from privacy_integration.data.dataset import CarlaFrameDataset, discover_town_dirs
from privacy_integration.data.splits import DataSplit, stratified_split
from privacy_integration.settings import Settings, settings

DEFAULT_TOWNS = ("town01", "town02", "town03", "town04", "town05")


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True, help="Root holding town0X folders.")
    parser.add_argument("--towns", nargs="+", default=list(DEFAULT_TOWNS))
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Read labels from <labels-dir>/<town>/ (the poisoning detector's clean labels). "
             "Frames without a label there are skipped.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=None,
        help="Minimum bounding-box area in native px^2 for an object to count as present.",
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])


def load_dataset_and_split(args: argparse.Namespace) -> tuple[CarlaFrameDataset, DataSplit, Settings, int]:
    """Build the dataset and reproduce the deterministic split for a seed."""
    cfg = settings()
    seed = args.seed if args.seed is not None else cfg.privacy.random_seed

    dataset_cfg = cfg.dataset
    if getattr(args, "min_box_area", None) is not None:
        from dataclasses import replace

        dataset_cfg = replace(dataset_cfg, min_box_area_native=args.min_box_area)

    town_dirs = discover_town_dirs(args.data_dir, args.towns)
    if not town_dirs:
        raise SystemExit(f"no town folders found under {args.data_dir}")

    dataset = CarlaFrameDataset(
        town_dirs, dataset_cfg, labels_root=getattr(args, "labels_dir", None)
    )
    split = stratified_split(
        dataset.labels, dataset_cfg.train_fraction, dataset_cfg.val_fraction, seed
    )
    return dataset, split, cfg, seed
