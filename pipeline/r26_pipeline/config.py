"""Pipeline configuration: where the inputs are, which environment runs what, and run settings."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

COMPONENTS = ("privacy", "utility", "compliance")
COMPONENT_FOLDERS = {
    "privacy": "Privacy-Integration",
    "utility": "Utility-Evaluation",
    "compliance": "Risk-Compliance",
}
REQUIRED_SECRETS = ("R26_DATASET_AES_KEY", "R26_AUDIT_HMAC_KEY")
DEFAULT_TOWNS = ("town01", "town02", "town03", "town04", "town05")


class ConfigError(ValueError):
    """pipeline.yaml is missing something or holds an invalid value."""


@dataclass(frozen=True, slots=True)
class Environment:
    """How to start Python for one component."""

    component: str
    python: str | None = None
    conda: str | None = None
    conda_executable: str | None = None

    def command(self, *args: str) -> list[str]:
        if self.python:
            return [self.python, *args]
        if self.conda:
            executable = self.conda_executable or os.environ.get("CONDA_EXE") or shutil.which("conda")
            if not executable:
                raise ConfigError("conda environments are configured but conda was not found")
            return [executable, "run", "--no-capture-output", "-n", self.conda, "python", *args]
        import sys

        return [sys.executable, *args]

    def describe(self) -> str:
        if self.python:
            return f"python {self.python}"
        if self.conda:
            return f"conda env '{self.conda}'"
        return "current interpreter"


@dataclass(frozen=True, slots=True)
class Inputs:
    encrypted_dataset: Path
    calibration_source: Path | None
    kitti: Path | None


@dataclass(frozen=True, slots=True)
class PrivacySettings:
    seeds: tuple[int, ...] = (42, 43, 44)
    configs: tuple[str, ...] = ("no_dp", "config_a", "config_b", "config_c")
    device: str | None = None
    epochs: int | None = None
    shadow_epochs: int | None = None
    attack_epochs: int | None = None
    calibration_epsilon: float | None = None


@dataclass(frozen=True, slots=True)
class UtilitySettings:
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    workers: int = 8
    patience: int = 15
    device: str | None = None
    split_mode: str = "random"
    fid_limit: int = 1000
    fid_crop: str = "resize"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    source: Path
    repository: Path
    workspace: Path
    inputs: Inputs
    environments: dict[str, Environment]
    towns: tuple[str, ...] = DEFAULT_TOWNS
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    utility: UtilitySettings = field(default_factory=UtilitySettings)

    def component_dir(self, component: str) -> Path:
        return self.repository / COMPONENT_FOLDERS[component]


def _path(value, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _known(section: dict, allowed: set[str], name: str) -> dict:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {', '.join(sorted(unknown))}")
    return section


def load_config(path: Path) -> PipelineConfig:
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"{path} not found. Copy pipeline.example.yaml to pipeline.yaml and edit it.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent

    inputs = _section(raw, "inputs")
    encrypted = _path(inputs.get("encrypted_dataset"), base)
    if encrypted is None:
        raise ConfigError("inputs.encrypted_dataset is required")

    env_raw = _section(raw, "environments")
    conda_executable = env_raw.get("conda_executable")
    environments = {}
    for component in COMPONENTS:
        entry = env_raw.get(component) or {}
        if isinstance(entry, str):
            entry = {"conda": entry}
        _known(entry, {"python", "conda"}, f"environments.{component}")
        environments[component] = Environment(
            component, entry.get("python"), entry.get("conda"), conda_executable
        )

    privacy_raw = _known(_section(raw, "privacy"), set(PrivacySettings.__slots__), "privacy")
    utility_raw = _known(_section(raw, "utility"), set(UtilitySettings.__slots__), "utility")
    privacy = PrivacySettings(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in privacy_raw.items()
    })
    utility = UtilitySettings(**utility_raw)

    if not privacy.seeds:
        raise ConfigError("privacy.seeds must list at least one seed")
    if utility.split_mode not in ("random", "block"):
        raise ConfigError("utility.split_mode must be 'random' or 'block'")
    if utility.fid_crop not in ("resize", "center-square"):
        raise ConfigError("utility.fid_crop must be 'resize' or 'center-square'")

    repository = _path(raw.get("repository"), base) or base.parent
    workspace = _path(raw.get("workspace"), base)
    if workspace is None:
        raise ConfigError("workspace is required")
    if workspace == repository or repository in workspace.parents:
        raise ConfigError("workspace must be outside the repository so run data is never committed")

    return PipelineConfig(
        source=path,
        repository=repository,
        workspace=workspace,
        inputs=Inputs(
            encrypted, _path(inputs.get("calibration_source"), base), _path(inputs.get("kitti"), base)
        ),
        environments=environments,
        towns=tuple(raw.get("towns") or DEFAULT_TOWNS),
        privacy=privacy,
        utility=utility,
    )
