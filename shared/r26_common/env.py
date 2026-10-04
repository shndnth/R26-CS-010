"""The project .env file and the two keys every component uses."""

from __future__ import annotations

import os
from pathlib import Path

DATASET_KEY = "R26_DATASET_AES_KEY"
HMAC_KEY = "R26_AUDIT_HMAC_KEY"
ENV_FILE_VARIABLE = "R26_ENV_FILE"
AES_KEY_BYTES = 32


class MissingCredentialError(RuntimeError):
    """A required key is not available."""


def repository_root() -> Path | None:
    """The repository holding this package, when installed in editable mode."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "shared").is_dir() and (candidate / ".env.example").is_file():
            return candidate
    return None


def find_env_file(start: Path | None = None) -> Path | None:
    explicit = os.environ.get(ENV_FILE_VARIABLE)
    if explicit:
        return Path(explicit)
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / ".env").is_file():
            return folder / ".env"
    root = repository_root()
    if root is not None and (root / ".env").is_file():
        return root / ".env"
    return None


def load_env(path: Path | None = None) -> Path | None:
    """Read KEY=VALUE lines into the environment without overriding it. Returns the file read."""
    path = path or find_env_file()
    if path is None or not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))
    return path


def require(name: str, purpose: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise MissingCredentialError(
            f"{name} is not set; it is needed for {purpose}. Copy .env.example at the repository "
            f"root to .env and fill it in, or set {name} in the environment. Never commit .env."
        )
    return value


def dataset_aes_key() -> bytes:
    key = require(DATASET_KEY, "decrypting the CARLA dataset").encode("utf-8")
    if len(key) != AES_KEY_BYTES:
        raise MissingCredentialError(f"{DATASET_KEY} must be exactly {AES_KEY_BYTES} bytes, got {len(key)}")
    return key


def audit_hmac_key() -> bytes:
    return require(HMAC_KEY, "signing and verifying audit reports").encode("utf-8")
