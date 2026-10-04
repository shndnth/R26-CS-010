"""Typed, validated project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from privacy_integration.paths import config_path, outputs_root


class ConfigError(ValueError):
    """Raised when config.yaml is missing keys or holds invalid values."""


@dataclass(frozen=True, slots=True)
class PrivacySettings:
    epsilon_total_max: float
    epsilon_calibration_max: float
    delta: float
    clipping_norm: float
    random_seed: int
    secure_rng: bool

    def __post_init__(self) -> None:
        if self.epsilon_calibration_max >= self.epsilon_total_max:
            raise ConfigError("epsilon_calibration_max must be below epsilon_total_max")
        if not 0 < self.delta < 1:
            raise ConfigError("delta must lie in (0, 1)")
        if self.clipping_norm <= 0:
            raise ConfigError("clipping_norm must be positive")


@dataclass(frozen=True, slots=True)
class AblationConfig:
    name: str
    target_epsilon: float
    epochs: int


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    batch_size: int
    feature_dim: int
    train_width: int
    train_height: int
    min_box_area_native: float
    native_width: int
    native_height: int
    train_fraction: float
    val_fraction: float

    def __post_init__(self) -> None:
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ConfigError("train_fraction + val_fraction must leave room for a test split")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class AttackSettings:
    n_shadow_models: int
    shadow_epochs: int
    attack_epochs: int
    success_rate_ceiling: float
    reid_ceiling: float


@dataclass(frozen=True, slots=True)
class Settings:
    privacy: PrivacySettings
    dataset: DatasetSettings
    attack: AttackSettings
    ablation: tuple[AblationConfig, ...]
    outputs_dir: Path = field(default_factory=outputs_root)

    def ablation_by_name(self, name: str) -> AblationConfig:
        for cfg in self.ablation:
            if cfg.name == name:
                return cfg
        available = ", ".join(c.name for c in self.ablation)
        raise ConfigError(f"unknown ablation config '{name}'; available: {available}")

    @property
    def ablation_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.ablation)


def _section(raw: dict, key: str) -> dict:
    if key not in raw:
        raise ConfigError(f"config.yaml is missing the '{key}' section")
    return raw[key]


def load_settings(path: Path | None = None) -> Settings:
    """Parse and validate config.yaml."""
    target = path or config_path()
    if not target.is_file():
        raise ConfigError(f"config.yaml not found at {target}")

    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    privacy_raw = _section(raw, "privacy")
    dataset_raw = _section(raw, "dataset")
    attack_raw = _section(raw, "attack")
    ablation_raw = _section(raw, "ablation").get("configs", [])

    if not ablation_raw:
        raise ConfigError("config.yaml defines no ablation configs")

    return Settings(
        privacy=PrivacySettings(
            epsilon_total_max=float(privacy_raw["epsilon_total_max"]),
            epsilon_calibration_max=float(privacy_raw["epsilon_calibration_max"]),
            delta=float(privacy_raw["delta"]),
            clipping_norm=float(privacy_raw["clipping_norm"]),
            random_seed=int(privacy_raw["random_seed"]),
            secure_rng=bool(privacy_raw.get("secure_rng", False)),
        ),
        dataset=DatasetSettings(
            batch_size=int(dataset_raw["batch_size"]),
            feature_dim=int(dataset_raw.get("feature_dim", 64)),
            train_width=int(dataset_raw.get("train_width", 448)),
            train_height=int(dataset_raw.get("train_height", 336)),
            min_box_area_native=float(dataset_raw.get("min_box_area_native", 1024.0)),
            native_width=int(dataset_raw.get("native_width", 800)),
            native_height=int(dataset_raw.get("native_height", 600)),
            train_fraction=float(dataset_raw.get("train_fraction", 0.70)),
            val_fraction=float(dataset_raw.get("val_fraction", 0.15)),
        ),
        attack=AttackSettings(
            n_shadow_models=int(attack_raw["n_shadow_models"]),
            shadow_epochs=int(attack_raw["shadow_epochs"]),
            attack_epochs=int(attack_raw["attack_epochs"]),
            success_rate_ceiling=float(attack_raw["success_rate_ceiling"]),
            reid_ceiling=float(attack_raw["reid_ceiling"]),
        ),
        ablation=tuple(
            AblationConfig(
                name=str(c["name"]),
                target_epsilon=float(c["target_epsilon"]),
                epochs=int(c["epochs"]),
            )
            for c in ablation_raw
        ),
    )


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings singleton."""
    return load_settings()
