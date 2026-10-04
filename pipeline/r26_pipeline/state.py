"""Run state and artefact lineage."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from r26_common.hashing import sha256_file

# Large files and folders are fingerprinted by size and mtime, not hashed.
CONTENT_HASH_LIMIT = 64 * 1024 * 1024
IGNORED_SUFFIXES = (".cache",)


def fingerprint(path: Path) -> str:
    """Content hash for normal files, a size and mtime digest for large files and directories."""
    if path.is_file():
        stat = path.stat()
        if stat.st_size <= CONTENT_HASH_LIMIT:
            return sha256_file(path)
        return "stat:" + hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        count = 0
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                if name.endswith(IGNORED_SUFFIXES):
                    continue
                full = Path(root) / name
                stat = full.stat()
                digest.update(f"{full.relative_to(path).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
                count += 1
        return f"dir:{count}:{digest.hexdigest()}"
    return "missing"


def code_fingerprint(component_dir: Path) -> str:
    """Content hash of a component's source and configuration, tests excluded."""
    digest = hashlib.sha256()
    for path in sorted(component_dir.rglob("*")):
        relative = path.relative_to(component_dir)
        if (not path.is_file() or path.suffix not in (".py", ".yaml", ".yml", ".toml")
                or relative.parts[0] in ("tests", "build") or "__pycache__" in relative.parts
                or any(part.endswith(".egg-info") for part in relative.parts)):
            continue
        digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def combined(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class StageRecord:
    status: str
    fingerprint: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    started: str | None = None
    finished: str | None = None
    seconds: float | None = None
    exit_code: int | None = None
    log: str | None = None
    detail: str | None = None
    code: str | None = None


class RunState:
    """pipeline_state.json in the workspace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stages: dict[str, StageRecord] = {}
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.stages = {name: StageRecord(**record) for name, record in raw.get("stages", {}).items()}

    def get(self, name: str) -> StageRecord | None:
        return self.stages.get(name)

    def set(self, name: str, record: StageRecord) -> None:
        self.stages[name] = record
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated": now(), "stages": {n: asdict(r) for n, r in self.stages.items()}}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
