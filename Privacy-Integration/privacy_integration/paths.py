"""Filesystem locations, resolved from the installed package rather than __file__ depth."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT_ENV_VAR = "R26_PROJECT_ROOT"
_OUTPUTS_ENV_VAR = "R26_OUTPUTS_DIR"


def project_root() -> Path:
    """Repository root."""
    override = os.environ.get(_ROOT_ENV_VAR)
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parent.parent


def outputs_root() -> Path:
    """Where calibration results, runs, figures and certificates are written."""
    override = os.environ.get(_OUTPUTS_ENV_VAR)
    return Path(override).resolve() if override else project_root() / "outputs"


def config_path() -> Path:
    return project_root() / "config.yaml"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
