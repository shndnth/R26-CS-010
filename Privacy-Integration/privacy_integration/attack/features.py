"""Attack signal extraction for membership inference."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

CONFIDENCE = 0
LOSS = 1
N_FEATURES = 2


@torch.no_grad()
def extract(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Extract the two-feature attack signal for every sample."""
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    was_training = model.training
    model.eval()

    confidences: list[float] = []
    losses: list[float] = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        confidences.extend(torch.sigmoid(logits).cpu().numpy().ravel())
        losses.extend(criterion(logits, labels).cpu().numpy().ravel())

    if was_training:
        model.train()

    return np.column_stack([confidences, losses])


def loss_gap(member_features: np.ndarray, nonmember_features: np.ndarray) -> float:
    """Mean non-member loss minus mean member loss."""
    return float(nonmember_features[:, LOSS].mean() - member_features[:, LOSS].mean())
