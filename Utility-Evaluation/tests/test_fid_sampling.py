"""FID image selection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from utility_evaluation.fid import center_square, sample_paths


def test_first_n_sorted_would_miss_later_towns():
    paths = [Path(f"town{t:02d}_{i:06d}.png") for t in range(1, 6) for i in range(300)]
    head = Counter(p.name[:6] for p in sorted(paths)[:1000])
    assert "town05" not in head


def test_seeded_sample_covers_every_town():
    paths = [Path(f"town{t:02d}_{i:06d}.png") for t in range(1, 6) for i in range(300)]
    sample = sample_paths(paths, 1000, seed=42)
    towns = Counter(p.name[:6] for p in sample)
    assert len(sample) == 1000
    assert set(towns) == {f"town{t:02d}" for t in range(1, 6)}
    assert sample == sample_paths(paths, 1000, seed=42)


def test_small_sets_are_used_whole():
    paths = [Path(f"{i}.png") for i in range(10)]
    assert sample_paths(paths, 1000, seed=1) == sorted(paths)


def test_center_square_crop():
    image = Image.new("RGB", (1242, 375))
    assert center_square(image).size == (375, 375)
