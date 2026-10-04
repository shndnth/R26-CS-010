"""AES-256-CBC decryption of the CARLA dataset and SHA-256 manifest verification."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from r26_common.hashing import sha256_hex

IV_LENGTH = 16


class IntegrityError(RuntimeError):
    """A decrypted file does not match its manifest digest."""


@dataclass(frozen=True, slots=True)
class DecryptionResult:
    decrypted: int
    failed: int
    verified: int
    unverified: int

    @property
    def total(self) -> int:
        return self.decrypted + self.failed


def decrypt_bytes(payload: bytes, key: bytes) -> bytes:
    if len(payload) <= IV_LENGTH:
        raise ValueError("payload shorter than the IV prefix")
    cipher = AES.new(key, AES.MODE_CBC, payload[:IV_LENGTH])
    return unpad(cipher.decrypt(payload[IV_LENGTH:]), AES.block_size)


def verify_digest(data: bytes, expected_hex: str) -> bool:
    """Constant-time comparison of a SHA-256 digest."""
    return hmac.compare_digest(sha256_hex(data), expected_hex)


def decrypt_file(source: Path, destination: Path, key: bytes, expected_hex: str | None) -> bool:
    """Decrypt one file."""
    plaintext = decrypt_bytes(source.read_bytes(), key)
    verified = False
    if expected_hex is not None:
        if not verify_digest(plaintext, expected_hex):
            raise IntegrityError(f"digest mismatch for {source.name}")
        verified = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plaintext)
    return verified
