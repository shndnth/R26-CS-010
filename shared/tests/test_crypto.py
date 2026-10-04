"""AES-256-CBC decryption and manifest verification."""

from __future__ import annotations

import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from r26_common.crypto import IntegrityError, decrypt_bytes, decrypt_file, verify_digest
from r26_common.hashing import sha256_file, sha256_hex

KEY = b"0123456789abcdef0123456789abcdef"
PLAINTEXT = b"\x89PNG\r\n\x1a\n" + b"synthetic frame payload" * 40


def encrypt(data: bytes, key: bytes = KEY) -> bytes:
    """Mirror of the Dataset Generation encryption."""
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(data, AES.block_size))


class TestDecryptBytes:
    def test_round_trip(self):
        assert decrypt_bytes(encrypt(PLAINTEXT), KEY) == PLAINTEXT

    def test_distinct_ivs_give_distinct_ciphertexts(self):
        assert encrypt(PLAINTEXT) != encrypt(PLAINTEXT)

    def test_wrong_key_does_not_recover_plaintext(self):
        try:
            recovered = decrypt_bytes(encrypt(PLAINTEXT), b"f" * 32)
        except ValueError:
            return  # padding check rejected the wrong key, the usual outcome
        assert recovered != PLAINTEXT

    def test_truncated_payload_is_rejected(self):
        with pytest.raises(ValueError, match="IV"):
            decrypt_bytes(b"short", KEY)


class TestDigests:
    def test_matching_digest(self):
        assert verify_digest(PLAINTEXT, sha256_hex(PLAINTEXT))

    def test_altered_content(self):
        assert not verify_digest(PLAINTEXT + b"x", sha256_hex(PLAINTEXT))

    def test_file_digest_matches_bytes_digest(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(PLAINTEXT)
        assert sha256_file(path) == sha256_hex(PLAINTEXT)


class TestDecryptFile:
    def test_writes_plaintext_and_reports_verification(self, tmp_path):
        source = tmp_path / "frame.png.enc"
        source.write_bytes(encrypt(PLAINTEXT))
        destination = tmp_path / "out" / "nested" / "frame.png"
        assert decrypt_file(source, destination, KEY, sha256_hex(PLAINTEXT)) is True
        assert destination.read_bytes() == PLAINTEXT

    def test_source_is_never_removed(self, tmp_path):
        source = tmp_path / "frame.png.enc"
        source.write_bytes(encrypt(PLAINTEXT))
        decrypt_file(source, tmp_path / "out.png", KEY, None)
        assert source.is_file()

    def test_tampered_payload_raises_and_writes_nothing(self, tmp_path):
        source = tmp_path / "frame.png.enc"
        source.write_bytes(encrypt(b"tampered content"))
        destination = tmp_path / "out.png"
        with pytest.raises(IntegrityError, match="digest mismatch"):
            decrypt_file(source, destination, KEY, sha256_hex(PLAINTEXT))
        assert not destination.exists()

    def test_absent_digest_skips_verification(self, tmp_path):
        source = tmp_path / "frame.png.enc"
        source.write_bytes(encrypt(PLAINTEXT))
        assert decrypt_file(source, tmp_path / "out.png", KEY, None) is False
