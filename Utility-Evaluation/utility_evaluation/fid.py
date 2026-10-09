"""Image selection and preprocessing for FID."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def sample_paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    """A seeded random sample of at most ``limit`` paths."""
    ordered = sorted(paths)
    if len(ordered) <= limit:
        return ordered
    return sorted(random.Random(seed).sample(ordered, limit))


def center_square(image: Image.Image) -> Image.Image:
    """Crop the central square so resizing to 299x299 does not distort the image."""
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return image.crop((left, top, left + side, top + side))
