"""Reference-calibrated membership inference (offline, fixed variance).

The plain loss-threshold attack treats every frame alike, so a frame that is
simply easy (low loss for any model) looks like a member. Calibration removes
that: a few reference models that never saw the attacked frames estimate how
easy each frame is, and the attack scores how much better the target does on a
frame than the references expect. This is the idea behind the offline
likelihood ratio attack of Carlini et al. (2022) and the difficulty
calibration of Watson et al. (2022), with a small number of reference models
and one shared variance, so it is cheaper than full LiRA and can find
individual members at a low false-positive rate.

Scores use the logit of the probability given to the true label. For this
binary classifier that is the raw logit with its sign flipped for negatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from privacy_integration.attack.threshold import LOW_FPRS, tpr_at_fpr
from privacy_integration.data.splits import balance_by_oversampling

MIN_SIGMA = 1e-3


@torch.no_grad()
def true_label_logits(
    model: nn.Module, dataset: Dataset, indices: Sequence[int], batch_size: int, device: str
) -> np.ndarray:
    """Logit of the probability the model gives to each frame's true label."""
    was_training = model.training
    model.eval()
    out: list[np.ndarray] = []
    for inputs, labels in DataLoader(Subset(dataset, list(indices)), batch_size=batch_size):
        logits = model(inputs.to(device)).view(-1).cpu().numpy()
        sign = 2.0 * labels.view(-1).numpy() - 1.0
        out.append(sign * logits)
    if was_training:
        model.train()
    return np.concatenate(out) if out else np.zeros(0)


def reference_pool(train_idx: Sequence[int], val_idx: Sequence[int], exclude: Sequence[int]) -> list[int]:
    """Frames the references may train on: never a frame that is attacked."""
    banned = set(exclude)
    return [i for i in [*train_idx, *val_idx] if i not in banned]


def reference_members(pool: Sequence[int], labels: Sequence[int], n: int, seed: int) -> list[int]:
    """A random training set for one reference model, balanced like the targets'."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(pool), size=min(n, len(pool)), replace=False).tolist()
    return balance_by_oversampling(chosen, labels)


def calibrated_scores(target: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Standardised gap between the target and the references, per frame.

    `target` has one value per frame; `references` has one row per reference
    model. Higher means more likely a member. One variance is shared by all
    frames, which is the stable choice when there are only a few references.
    """
    refs = np.atleast_2d(references)
    mu = refs.mean(axis=0)
    if refs.shape[0] > 1:
        sigma = float(np.sqrt(np.mean(refs.var(axis=0, ddof=1))))
    else:
        sigma = float(refs.std())
    return (target - mu) / max(sigma, MIN_SIGMA)


@dataclass(frozen=True, slots=True)
class CalibratedOutcome:
    auc: float
    tpr_at_fpr: dict[float, float]

    def as_dict(self) -> dict:
        return {
            "attack": "reference_calibrated",
            "attack_auc": round(self.auc, 4),
            **{f"tpr_at_{int(f * 100)}pct_fpr": round(t, 4) for f, t in self.tpr_at_fpr.items()},
        }


def evaluate_scores(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> CalibratedOutcome:
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    auc = float(roc_auc_score(labels, np.concatenate([member_scores, nonmember_scores])))
    return CalibratedOutcome(
        auc=auc, tpr_at_fpr={f: tpr_at_fpr(member_scores, nonmember_scores, f) for f in LOW_FPRS}
    )
