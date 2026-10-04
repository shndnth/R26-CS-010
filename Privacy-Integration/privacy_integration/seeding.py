"""Deterministic seeding across every RNG the pipeline touches."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generator(seed: int) -> torch.Generator:
    """Seeded torch Generator for DataLoader shuffling and splits."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
