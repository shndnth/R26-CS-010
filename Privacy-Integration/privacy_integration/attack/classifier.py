"""Binary attack classifier over the two-feature membership signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from privacy_integration.attack.features import N_FEATURES

RANDOM_BASELINE_PCT = 50.0


@dataclass(frozen=True, slots=True)
class AttackOutcome:
    success_rate_pct: float
    auc: float
    interpretation: str
    n_members: int
    n_nonmembers: int
    member_loss_mean: float
    nonmember_loss_mean: float
    loss_gap: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "attack_success_rate_pct": round(self.success_rate_pct, 2),
            "attack_auc": round(self.auc, 4),
            "interpretation": self.interpretation,
            "n_members": self.n_members,
            "n_nonmembers": self.n_nonmembers,
            "target_member_loss_mean": round(self.member_loss_mean, 4),
            "target_nonmember_loss_mean": round(self.nonmember_loss_mean, 4),
            "target_loss_gap": round(self.loss_gap, 4),
        }


def build_classifier(device: str, hidden: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_FEATURES, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(inplace=True),
        nn.Linear(hidden // 2, 1),
    ).to(device)


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    device: str,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> nn.Module:
    inputs = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True)

    model = build_classifier(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            criterion(model(batch_x), batch_y).backward()
            optimizer.step()
    return model


def interpret(success_rate_pct: float, ceiling_pct: float) -> str:
    if success_rate_pct <= 52.0:
        return "Near random: strong privacy protection"
    if success_rate_pct <= ceiling_pct:
        return "Within target: privacy guarantee holds"
    if success_rate_pct <= 65.0:
        return "Above target: privacy weakened"
    return "High attack success: formal guarantee not translating"


@torch.no_grad()
def evaluate_attack(
    classifier: nn.Module,
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    device: str,
    ceiling_pct: float,
) -> AttackOutcome:
    """Score the attack against a target model."""
    features = np.vstack([member_features, nonmember_features])
    truth = np.concatenate(
        [np.ones(len(member_features), dtype=int), np.zeros(len(nonmember_features), dtype=int)]
    )

    classifier.eval()
    logits = classifier(torch.tensor(features, dtype=torch.float32).to(device))
    logits = logits.cpu().numpy().ravel()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (logits > 0).astype(int)

    success_rate = 100.0 * float(np.mean(predictions == truth))
    try:
        auc = float(roc_auc_score(truth, probabilities))
    except ValueError:
        auc = float("nan")

    return AttackOutcome(
        success_rate_pct=success_rate,
        auc=auc,
        interpretation=interpret(success_rate, ceiling_pct),
        n_members=len(member_features),
        n_nonmembers=len(nonmember_features),
        member_loss_mean=float(member_features[:, 1].mean()),
        nonmember_loss_mean=float(nonmember_features[:, 1].mean()),
        loss_gap=float(nonmember_features[:, 1].mean() - member_features[:, 1].mean()),
    )
