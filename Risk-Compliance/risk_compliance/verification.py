"""Independent verification of signed audit certificates."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path

ALGORITHM = "HMAC-SHA256"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    report: str
    algorithm: str | None
    signature_valid: bool
    digest_valid: bool

    @property
    def passed(self) -> bool:
        return self.signature_valid and self.digest_valid

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict:
        return {
            "report": self.report,
            "algorithm": self.algorithm,
            "signature_valid": self.signature_valid,
            "digest_valid": self.digest_valid,
            "verdict": self.verdict,
        }


def compute_signature(payload: str, key: bytes) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify(report_path: Path, certificate_path: Path, key: bytes) -> VerificationResult:
    """Check a report against its certificate."""
    payload = report_path.read_text(encoding="utf-8")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))

    signature_valid = hmac.compare_digest(
        str(certificate.get("hmac_signature", "")),
        compute_signature(payload, key),
    )
    digest_valid = hmac.compare_digest(
        str(certificate.get("sha256_content", "")),
        compute_digest(payload),
    )

    return VerificationResult(
        report=report_path.name,
        algorithm=certificate.get("algorithm"),
        signature_valid=signature_valid,
        digest_valid=digest_valid,
    )


def verify_all(runs_root: Path, key: bytes) -> list[VerificationResult]:
    """Verify every audit report found under a runs directory."""
    results: list[VerificationResult] = []
    for report_path in sorted(runs_root.rglob("audit_report.json")):
        certificate_path = report_path.with_name("audit_certificate.json")
        if certificate_path.is_file():
            results.append(verify(report_path, certificate_path, key))
    return results


def tamper_test(
    report_path: Path,
    certificate_path: Path,
    key: bytes,
    workspace: Path,
    field_path: tuple[str, ...] = ("results", "final_epsilon"),
    replacement: object = 0.01,
) -> tuple[object, VerificationResult]:
    """Falsify a copied report and confirm verification rejects it."""
    import shutil

    workspace.mkdir(parents=True, exist_ok=True)
    tampered_report = workspace / report_path.name
    tampered_certificate = workspace / certificate_path.name
    shutil.copy2(report_path, tampered_report)
    shutil.copy2(certificate_path, tampered_certificate)

    payload = json.loads(tampered_report.read_text(encoding="utf-8"))
    target = payload
    for key_name in field_path[:-1]:
        target = target[key_name]
    original_value = target[field_path[-1]]
    target[field_path[-1]] = replacement

    tampered_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = verify(tampered_report, tampered_certificate, key)
    return original_value, result
