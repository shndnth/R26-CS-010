"""Orchestration for a single training experiment, private or non-private."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from privacy_integration.artifacts import RunPaths
from privacy_integration.audit.report import PrivacyAuditInterface
from privacy_integration.audit.signing import CertificateSigner
from privacy_integration.data.dataset import CarlaFrameDataset
from privacy_integration.data.splits import DataSplit, balance_by_oversampling
from privacy_integration.evaluation.metrics import ClassificationMetrics, evaluate
from privacy_integration.logging_config import get_logger
from privacy_integration.models.cnn import PresenceClassifier
from privacy_integration.serialization import utc_timestamp, write_json
from privacy_integration.settings import Settings
from privacy_integration.training.dp_trainer import DPTrainingWrapper

logger = get_logger(__name__)

# SGD with momentum: Adam's second moment accumulated the DP noise and destabilised training.
DP_DEFAULT_LR = 1e-3
BASELINE_DEFAULT_LR = 1e-4
MOMENTUM = 0.9


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    tag: str
    seed: int
    epsilon: float | None
    final_epsilon: float | None
    noise_multiplier: float | None
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics

    def as_dict(self) -> dict:
        return {
            "config": self.tag,
            "seed": self.seed,
            "target_epsilon": self.epsilon,
            "final_epsilon": self.final_epsilon,
            "noise_multiplier": self.noise_multiplier,
            "val_metrics": self.val_metrics.as_dict(),
            "test_metrics": self.test_metrics.as_dict(),
        }


def build_loaders(
    dataset: CarlaFrameDataset, split: DataSplit, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Training loader is class-balanced; validation and test keep the natural prior."""
    balanced = balance_by_oversampling(split.train, dataset.labels)
    logger.info(
        "Train split rebalanced | %d -> %d samples (val=%d test=%d)",
        len(split.train), len(balanced), len(split.val), len(split.test),
    )
    return (
        DataLoader(Subset(dataset, balanced), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(dataset, split.val), batch_size=batch_size),
        DataLoader(Subset(dataset, split.test), batch_size=batch_size),
    )


def train_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    """Non-private baseline establishing the utility ceiling."""
    model = PresenceClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Baseline training | epochs=%d lr=%s device=%s", epochs, lr, device)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, batches = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        mean_loss = total_loss / max(batches, 1)
        if val_loader is None:
            logger.info("Baseline epoch %d/%d | train_loss=%.4f", epoch, epochs, mean_loss)
        else:
            metrics = evaluate(model, val_loader, device)
            logger.info(
                "Baseline epoch %d/%d | train_loss=%.4f | %s",
                epoch, epochs, mean_loss, metrics.summary(),
            )
    return model


def train_private(
    target_epsilon: float,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    settings: Settings,
) -> tuple[nn.Module, object, float]:
    """DP-SGD training for one ablation configuration."""
    model = PresenceClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    wrapper = DPTrainingWrapper(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        epochs=epochs,
        epsilon_max=settings.privacy.epsilon_total_max,
        delta=settings.privacy.delta,
        clipping_norm=settings.privacy.clipping_norm,
        device=device,
    )
    result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=True)
    return wrapper.model, result, wrapper.noise_multiplier


def persist(
    paths: RunPaths,
    model: nn.Module,
    result: ExperimentResult,
    run_meta: dict,
    training_result: object | None,
    wrapper: DPTrainingWrapper | None,
    settings: Settings,
) -> None:
    """Write checkpoint, metrics, and for private runs the signed audit trail."""
    paths.create()

    torch.save(
        {"state_dict": model.state_dict(), "meta": run_meta, **result.as_dict()},
        paths.checkpoint,
    )
    write_json({**run_meta, **result.as_dict()}, paths.metrics)

    if wrapper is None or training_result is None:
        _write_non_private_declaration(paths, result, run_meta)
        logger.info(
            "%s | checkpoint, metrics and signed non-private declaration written to %s",
            result.tag, paths.run_dir,
        )
        return

    wrapper.save_budget_log(training_result, paths.budget_log)

    audit = PrivacyAuditInterface()
    report = audit.build_report(
        result=training_result,
        clipping_norm=settings.privacy.clipping_norm,
        noise_multiplier=result.noise_multiplier,
        delta=settings.privacy.delta,
        epsilon_max=settings.privacy.epsilon_total_max,
    )
    audit.save_report(report, paths.audit_report)
    CertificateSigner().sign(paths.audit_report, paths.audit_certificate)

    logger.info("%s | checkpoint, budget log, signed audit written to %s", result.tag, paths.run_dir)


def _write_non_private_declaration(paths: RunPaths, result: ExperimentResult, run_meta: dict) -> None:
    """Record an explicit statement that no privacy mechanism was applied."""
    declaration = {
        "differential_privacy_applied": False,
        "project": "R26-CS-010",
        "member": "IT22309556",
        "config": result.tag,
        "seed": result.seed,
        "generated_at": utc_timestamp(),
        "privacy_config": {
            "epsilon": None,
            "delta": None,
            "clipping_norm": None,
            "noise_multiplier": None,
        },
        "results": {
            "budget_log_written": False,
            "budget_respected": None,
            "halted_early": False,
        },
        "rationale": (
            "Non-private control run. No differential privacy mechanism was applied, "
            "so no privacy budget exists to account for, report or enforce. Included "
            "as the unprotected baseline against which the DP configurations are "
            "compared, and as the positive control demonstrating that the membership "
            "inference attack succeeds when no protection is present."
        ),
        "warning": (
            "The absence of an epsilon value is intentional and MUST NOT be read as "
            "epsilon = 0. This run carries no privacy guarantee whatsoever."
        ),
        "run_metadata": run_meta,
    }

    write_json(declaration, paths.audit_report)
    CertificateSigner().sign(paths.audit_report, paths.audit_certificate)


def load_checkpoint_model(path: Path, device: str) -> tuple[nn.Module, dict]:
    """Restore a trained model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = {
        key.removeprefix("_module."): value for key, value in checkpoint["state_dict"].items()
    }
    model = PresenceClassifier()
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint
