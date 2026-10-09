"""Shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("R26_DATASET_AES_KEY", "0123456789abcdef0123456789abcdef")


@pytest.fixture
def label_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "labels"
    directory.mkdir()
    return directory


def write_label(directory: Path, name: str, lines: list[str]) -> Path:
    path = directory / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
