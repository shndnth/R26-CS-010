"""Decrypt one town of the CARLA delivery and verify it against its manifest."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from r26_common.crypto import IntegrityError, decrypt_file


def load_manifest(town_dir: Path) -> dict[str, str]:
    """The town's own ``<town>_manifest.json``."""
    path = town_dir / f"{town_dir.name}_manifest.json"
    if not path.is_file():
        found = sorted(town_dir.glob("*_manifest.json"))
        if len(found) != 1:
            return {}
        path = found[0]
    return json.loads(path.read_text(encoding="utf-8"))


def decrypt_town(town_dir: Path, out_dir: Path, key: bytes,
                 say: Callable[[str], None] = print) -> dict[str, int]:
    """Decrypt every ``.enc`` file, copy labels and metadata. Sources are never changed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(town_dir)
    if manifest:
        say(f"Manifest loaded for {town_dir.name} ({len(manifest)} entries)")
    else:
        say(f"WARNING: no {town_dir.name}_manifest.json, files cannot be hash-verified")

    tally = {"verified": 0, "unverified": 0, "failed": 0}
    for source in sorted(town_dir.rglob("*.enc")):
        original = source.name.removesuffix(".enc")
        destination = out_dir / source.parent.relative_to(town_dir) / original
        try:
            verified = decrypt_file(source, destination, key, manifest.get(original))
        except IntegrityError:
            say(f"HASH MISMATCH: {original}, possible corruption or tampering")
            tally["failed"] += 1
            continue
        except ValueError as exc:
            say(f"Failed to decrypt {source.name}: {exc}")
            tally["failed"] += 1
            continue
        tally["verified" if verified else "unverified"] += 1

    labels = town_dir / "labels"
    if labels.is_dir():
        shutil.copytree(labels, out_dir / "labels", dirs_exist_ok=True)
        say(f"Copied labels/ ({sum(1 for _ in labels.iterdir())} files, plain text)")

    for metadata in town_dir.iterdir():
        if metadata.is_file() and metadata.suffix in (".json", ".csv"):
            shutil.copy2(metadata, out_dir)

    say(f"{town_dir.name}: {tally['verified']} verified, {tally['unverified']} decrypted "
        f"without a manifest entry, {tally['failed']} failed\n")
    return tally
