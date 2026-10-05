"""Loss-threshold membership inference (Yeom et al., 2018) with low-FPR reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

LOW_FPRS = (0.01, 0.05)


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr: float) -> float:
    """True-positive rate at the largest threshold whose false-positive rate is at most `fpr`."""
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    fprs, tprs, _ = roc_curve(labels, scores)
    allowed = fprs <= fpr
    return float(tprs[allowed].max()) if allowed.any() else 0.0


@dataclass(frozen=True)
class ThresholdOutcome:
    auc: float
    tpr_at_fpr: dict[float, float]
    member_loss_mean: float
    nonmember_loss_mean: float

    def as_dict(self) -> dict:
        return {
            "attack": "loss_threshold",
            "attack_auc": round(self.auc, 4),
            **{f"tpr_at_{int(f * 100)}pct_fpr": round(t, 4) for f, t in self.tpr_at_fpr.items()},
            **{f"random_tpr_at_{int(f * 100)}pct_fpr": f for f in self.tpr_at_fpr},
            "member_loss_mean": round(self.member_loss_mean, 4),
            "nonmember_loss_mean": round(self.nonmember_loss_mean, 4),
        }


def loss_threshold_attack(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> ThresholdOutcome:
    """Score each sample by its negative loss: lower loss means more likely a member."""
    member_scores, nonmember_scores = -np.asarray(member_losses), -np.asarray(nonmember_losses)
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    auc = float(roc_auc_score(labels, np.concatenate([member_scores, nonmember_scores])))
    return ThresholdOutcome(
        auc=auc,
        tpr_at_fpr={f: tpr_at_fpr(member_scores, nonmember_scores, f) for f in LOW_FPRS},
        member_loss_mean=float(np.mean(member_losses)),
        nonmember_loss_mean=float(np.mean(nonmember_losses)),
    )
