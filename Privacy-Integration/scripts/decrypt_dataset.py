"""Decrypt the AES-256 encrypted CARLA dataset and verify manifest integrity."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from r26_common.crypto import DecryptionResult, IntegrityError, decrypt_file
from r26_common.env import dataset_aes_key

from privacy_integration.logging_config import get_logger
from privacy_integration.serialization import read_json
from scripts._common import DEFAULT_TOWNS

logger = get_logger("decrypt")

PROGRESS_INTERVAL = 500


def decrypt_town(town_dir: Path, output_dir: Path, key: bytes) -> DecryptionResult:
    manifest_path = town_dir / f"{town_dir.name}_manifest.json"
    manifest: dict[str, str] = {}
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        logger.info("%s | manifest loaded (%d digests)", town_dir.name, len(manifest))
    else:
        logger.warning("%s | no manifest; integrity cannot be verified", town_dir.name)

    encrypted = sorted(town_dir.rglob("*.enc"))
    decrypted = failed = verified = unverified = 0
    started = time.monotonic()

    for index, source in enumerate(encrypted, start=1):
        original_name = source.name.removesuffix(".enc")
        destination = output_dir / source.parent.relative_to(town_dir) / original_name
        try:
            was_verified = decrypt_file(source, destination, key, manifest.get(original_name))
        except (IntegrityError, ValueError) as exc:
            logger.error("%s | %s", source.name, exc)
            failed += 1
            continue

        decrypted += 1
        verified += int(was_verified)
        unverified += int(not was_verified)

        if index % PROGRESS_INTERVAL == 0 or index == len(encrypted):
            logger.info(
                "%s | %d/%d (%.0fs)", town_dir.name, index, len(encrypted), time.monotonic() - started
            )

    return DecryptionResult(decrypted, failed, verified, unverified)


def copy_labels(town_dir: Path, output_dir: Path) -> int:
    """Copy unencrypted YOLO label files verbatim."""
    labels_dir = town_dir / "labels"
    if not labels_dir.is_dir():
        return 0

    target = output_dir / "labels"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in labels_dir.glob("*.txt"):
        (target / source.name).write_bytes(source.read_bytes())
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decrypt the CARLA dataset handoff")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--towns", nargs="+", default=list(DEFAULT_TOWNS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = dataset_aes_key()

    totals = DecryptionResult(0, 0, 0, 0)
    total_labels = 0

    for town in args.towns:
        town_dir = args.input_dir / town
        if not town_dir.is_dir():
            logger.warning("skipping %s | not found", town)
            continue

        logger.info("decrypting %s", town)
        result = decrypt_town(town_dir, args.output_dir / town, key)
        labels = copy_labels(town_dir, args.output_dir / town)
        total_labels += labels

        totals = DecryptionResult(
            totals.decrypted + result.decrypted,
            totals.failed + result.failed,
            totals.verified + result.verified,
            totals.unverified + result.unverified,
        )
        logger.info(
            "%s | decrypted=%d failed=%d verified=%d labels=%d",
            town, result.decrypted, result.failed, result.verified, labels,
        )

    logger.info(
        "complete | decrypted=%d failed=%d hash-verified=%d unverified=%d labels=%d",
        totals.decrypted, totals.failed, totals.verified, totals.unverified, total_labels,
    )
    if totals.failed:
        raise SystemExit(f"{totals.failed} file(s) failed decryption or integrity check")


if __name__ == "__main__":
    main()
