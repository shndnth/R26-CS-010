from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from privacy_integration.legacy_constants import CLIPPING_NORM, DELTA, EPSILON_MAX
from privacy_integration.logging_config import get_logger as setup_logging
from privacy_integration.serialization import utc_timestamp as timestamp

logger = setup_logging(__name__)


class BudgetExceededError(Exception):
    pass


@dataclass
class StepRecord:
    step: int
    epoch: int
    cumulative_eps: float
    loss: float


@dataclass
class TrainingResult:
    budget_log: list[StepRecord] = field(default_factory=list)
    final_epsilon: float = 0.0
    noise_multiplier: float = 0.0
    total_steps: int = 0
    halted_early: bool = False
    generated_at: str = field(default_factory=timestamp)


class DPTrainingWrapper:
    """Opacus DP-SGD: per-sample gradient clipping and Gaussian noise."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        epochs: int,
        epsilon_max: float = EPSILON_MAX,
        delta: float = DELTA,
        clipping_norm: float = CLIPPING_NORM,
        noise_multiplier: float | None = None,
        device: str | None = None,
    ) -> None:
        self._epsilon_max = epsilon_max
        self._delta = delta
        self._clipping_norm = clipping_norm
        self._epochs = epochs
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not ModuleValidator.is_valid(model):
            logger.warning("Unsupported layers detected. Applying ModuleValidator.fix()")
            model = ModuleValidator.fix(model)

        self._model = model.to(self._device)
        # secure_mode off: torchcsprng is unmaintained and only supports PyTorch 1.8.1.
        engine = PrivacyEngine(secure_mode=False)

        if noise_multiplier is not None:
            self._model, self._optimizer, self._loader = engine.make_private(
                module=self._model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=clipping_norm,
            )
            self._noise_multiplier = noise_multiplier
        else:
            self._model, self._optimizer, self._loader = engine.make_private_with_epsilon(
                module=self._model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=delta,
                max_grad_norm=clipping_norm,
            )
            self._noise_multiplier = self._optimizer.noise_multiplier

        self._engine = engine
        self._result = TrainingResult(noise_multiplier=self._noise_multiplier)

        logger.info(
            "Initialised | C=%.4f sigma=%.4f eps_max=%.1f device=%s",
            clipping_norm, self._noise_multiplier, epsilon_max, self._device,
        )

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def noise_multiplier(self) -> float:
        return self._noise_multiplier

    def _current_epsilon(self) -> float:
        return self._engine.get_epsilon(delta=self._delta)

    def _check_budget(self) -> float:
        eps = self._current_epsilon()
        if eps >= self._epsilon_max:
            self._result.halted_early = True
            raise BudgetExceededError(
                f"Budget ceiling reached: eps={eps:.4f} >= {self._epsilon_max}"
            )
        return eps

    def _step(self, loss: torch.Tensor, epoch: int) -> float:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        eps = self._check_budget()
        self._result.budget_log.append(
            StepRecord(
                step=self._result.total_steps,
                epoch=epoch,
                cumulative_eps=round(eps, 6),
                loss=round(loss.item(), 6),
            )
        )
        self._result.total_steps += 1
        return eps

    def train(
        self,
        criterion: nn.Module,
        label_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        verbose: bool = True,
    ) -> TrainingResult:
        """Run the full training loop."""
        self._model.train()
        logger.info("Training started | epochs=%d", self._epochs)

        try:
            for epoch in range(self._epochs):
                total_loss, batches = 0.0, 0

                for batch in self._loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        X, y = batch
                    else:
                        X = batch[0] if isinstance(batch, (list, tuple)) else batch
                        y = label_fn(X) if label_fn else None

                    X = X.to(self._device)
                    output = self._model(X)

                    if y is not None:
                        y = y.to(self._device)
                        loss = criterion(output, y)
                    else:
                        loss = criterion(output)

                    eps = self._step(loss, epoch)
                    total_loss += loss.item()
                    batches += 1

                if verbose:
                    logger.info(
                        "Epoch %d/%d | loss=%.4f | eps=%.4f",
                        epoch + 1, self._epochs,
                        total_loss / batches, eps,
                    )

        except BudgetExceededError as exc:
            logger.warning("Budget halt: %s", exc)

        self._result.final_epsilon = self._current_epsilon()
        self._result.generated_at = timestamp()

        logger.info(
            "Training complete | steps=%d eps=%.4f halted=%s",
            self._result.total_steps,
            self._result.final_epsilon,
            self._result.halted_early,
        )

        return self._result

    def save_budget_log(self, result: TrainingResult, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {"step": r.step, "epoch": r.epoch,
                 "cumulative_eps": r.cumulative_eps, "loss": r.loss}
                for r in result.budget_log
            ]
        )
        df.to_csv(path, index=False)
        logger.info("Budget log saved -> %s (%d rows)", path, len(df))
