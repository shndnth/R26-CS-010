"""Convolutional classifier for pedestrian/cyclist presence."""

from __future__ import annotations

import torch
from torch import nn


class PresenceClassifier(nn.Module):
    """Binary presence classifier, DP-compatible by construction."""

    def __init__(self, in_channels: int = 3, width: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.features = nn.Sequential(
            *self._block(in_channels, c1, stride=2),
            *self._block(c1, c2, stride=2),
            *self._block(c2, c3, stride=2),
            *self._block(c3, c3, stride=1),
            nn.AdaptiveMaxPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, c3 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c3 // 2, 1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int, stride: int) -> tuple[nn.Module, ...]:
        groups = 8 if out_ch % 8 == 0 else 1
        return (
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
