"""Typed configuration and credential loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parent.parent


class ConfigError(ValueError):
    """Raised when config.yaml is missing keys or holds invalid values."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    epsilon_ceiling: float
    mia_success_ceiling: float
    reid_ceiling: float
    epsilon_agreement_tolerance: float


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    epsilon: float
    sensitivities: dict[str, float]

    def laplace_scale(self, field: str) -> float:
        """Scale parameter b = sensitivity / epsilon."""
        return self.sensitivities[field] / self.epsilon


@dataclass(frozen=True, slots=True)
class AnonymitySettings:
    quasi_identifiers: tuple[str, ...]
    k_threshold: int


@dataclass(frozen=True, slots=True)
class LeakageSettings:
    scan_suffixes: frozenset[str]


@dataclass(frozen=True, slots=True)
class Settings:
    thresholds: Thresholds
    calibration: CalibrationSettings
    anonymity: AnonymitySettings
    leakage: LeakageSettings


def load_settings(path: Path | None = None) -> Settings:
    target = path or project_root() / "config.yaml"
    if not target.is_file():
        raise ConfigError(f"config.yaml not found at {target}")

    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    th, cal = raw["thresholds"], raw["calibration"]
    anon, leak = raw["anonymity"], raw["leakage"]

    return Settings(
        thresholds=Thresholds(
            epsilon_ceiling=float(th["epsilon_ceiling"]),
            mia_success_ceiling=float(th["mia_success_ceiling"]),
            reid_ceiling=float(th["reid_ceiling"]),
            epsilon_agreement_tolerance=float(th["epsilon_agreement_tolerance"]),
        ),
        calibration=CalibrationSettings(
            epsilon=float(cal["epsilon"]),
            sensitivities={k: float(v) for k, v in cal["sensitivities"].items()},
        ),
        anonymity=AnonymitySettings(
            quasi_identifiers=tuple(anon["quasi_identifiers"]),
            k_threshold=int(anon["k_threshold"]),
        ),
        leakage=LeakageSettings(
            scan_suffixes=frozenset(leak["scan_suffixes"]),
        ),
    )


@lru_cache(maxsize=1)
def settings() -> Settings:
    return load_settings()
