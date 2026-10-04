"""Classification metrics for imbalanced binary evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Held-out performance."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"loss={self.loss:.4f} acc={self.accuracy:.2f}% prec={self.precision:.2f}% "
            f"rec={self.recall:.2f}% f1={self.f1:.2f}% auc={self.auc:.4f}"
        )


def compute_metrics(
    probabilities: np.ndarray, targets: np.ndarray, loss: float = 0.0, threshold: float = 0.5
) -> ClassificationMetrics:
    predictions = (probabilities > threshold).astype(int)
    targets = targets.astype(int)

    tp = int(np.sum((predictions == 1) & (targets == 1)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))

    total = max(tp + fp + fn + tn, 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    try:
        auc = float(roc_auc_score(targets, probabilities))
    except ValueError:
        # Undefined when the evaluation set contains a single class.
        auc = float("nan")

    return ClassificationMetrics(
        loss=loss,
        accuracy=100.0 * (tp + tn) / total,
        precision=100.0 * precision,
        recall=100.0 * recall,
        f1=100.0 * f1,
        auc=auc,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> ClassificationMetrics:
    """Run the model over a loader and compute held-out metrics."""
    import torch
    from torch import nn

    criterion = nn.BCEWithLogitsLoss()
    was_training = model.training
    model.eval()

    probabilities: list[float] = []
    targets: list[float] = []
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item()
            batches += 1
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.extend(labels.cpu().numpy().ravel())

    if was_training:
        model.train()

    return compute_metrics(
        np.asarray(probabilities),
        np.asarray(targets),
        loss=total_loss / max(batches, 1),
    )
