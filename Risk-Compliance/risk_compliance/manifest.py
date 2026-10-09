"""Independent check of the decrypted dataset against the generator's manifests."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TownResult:
    town: str
    manifest_found: bool
    entries: int = 0
    verified: int = 0
    mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.manifest_found and not self.mismatched and not self.missing

    def as_dict(self) -> dict:
        return {
            "town": self.town,
            "manifest_found": self.manifest_found,
            "entries": self.entries,
            "verified": self.verified,
            "mismatched": len(self.mismatched),
            "missing": len(self.missing),
            "mismatched_files": self.mismatched[:50],
            "missing_files": self.missing[:50],
            "passed": self.passed,
        }


def verify_town(encrypted_town: Path, decrypted_town: Path) -> TownResult:
    manifest_path = encrypted_town / f"{encrypted_town.name}_manifest.json"
    result = TownResult(encrypted_town.name, manifest_path.is_file())
    if not result.manifest_found:
        return result

    manifest: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    located = {p.name: p for p in decrypted_town.rglob("*") if p.is_file()} if decrypted_town.is_dir() else {}
    result.entries = len(manifest)
    for name, expected in sorted(manifest.items()):
        path = located.get(name)
        if path is None:
            result.missing.append(name)
        elif hmac.compare_digest(sha256_file(path), expected):
            result.verified += 1
        else:
            result.mismatched.append(name)
    return result


def verify_dataset(
    encrypted_root: Path, decrypted_root: Path, towns: list[str] | None = None
) -> list[TownResult]:
    names = towns or sorted(p.name for p in encrypted_root.iterdir() if p.is_dir())
    return [
        verify_town(encrypted_root / t, decrypted_root / t)
        for t in names
        if (encrypted_root / t).is_dir()
    ]
