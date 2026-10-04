"""Tamper-evident signing of privacy audit reports."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from r26_common.env import audit_hmac_key

from privacy_integration.logging_config import get_logger
from privacy_integration.serialization import read_json, utc_timestamp, write_json

logger = get_logger(__name__)

ALGORITHM = "HMAC-SHA256"


class CertificateSigner:
    """Signs audit reports with HMAC-SHA256 so post-hoc modification is detectable."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key if key is not None else audit_hmac_key()

    def _signature(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def sign(self, report_path: Path, certificate_path: Path) -> dict[str, str]:
        payload = report_path.read_text(encoding="utf-8")
        signature = self._signature(payload)

        certificate = {
            "project": "R26-CS-010",
            "member": "IT22309556",
            "report_file": report_path.name,
            "sha256_content": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "hmac_signature": signature,
            "algorithm": ALGORITHM,
            "signed_at": utc_timestamp(),
            "purpose": (
                "Tamper-evident audit certificate. Any modification to the report "
                "after signing invalidates this signature. Supports GDPR Article 5(2) "
                "accountability."
            ),
        }
        write_json(certificate, certificate_path)
        logger.info("Signed %s -> %s (%s...)", report_path.name, certificate_path.name, signature[:16])
        return certificate

    def verify(self, report_path: Path, certificate_path: Path) -> bool:
        """Constant-time verification of both the digest and the signature."""
        payload = report_path.read_text(encoding="utf-8")
        certificate = read_json(certificate_path)

        signature_ok = hmac.compare_digest(
            str(certificate.get("hmac_signature", "")), self._signature(payload)
        )
        digest_ok = hmac.compare_digest(
            str(certificate.get("sha256_content", "")),
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )
        valid = signature_ok and digest_ok

        if valid:
            logger.info("Verification passed for %s", report_path.name)
        else:
            logger.warning("Verification FAILED for %s; report may be altered", report_path.name)
        return valid
