"""The compliance certificate must change visibly when the report or the PDF changes."""

from __future__ import annotations

import json
import sys

import pytest

from risk_compliance.certificate import (
    certificate_id,
    collect,
    qr_payload,
    render_pdf,
    sidecar,
    verify_certificate,
)

KEY = b"test-hmac-key-not-for-production"


def _report(tmp_path, verdict="PASS"):
    report = {
        "project": "R26-CS-010",
        "generated_at": "2026-10-18T10:00:00+00:00",
        "checks_run": ["certificate_verification", "budget_audit", "calibration_composition"],
        "checks_not_run": ["poisoning_detection"],
        "regulatory_mapping": [
            {"provision": "GDPR Art. 5(2)", "evidence_check": "certificate_verification", "verified": True},
            {"provision": "GDPR Art. 25", "evidence_check": "budget_audit", "verified": True},
            {"provision": "GDPR Art. 25", "evidence_check": "calibration_composition", "verified": False},
            {"provision": "GDPR Art. 32", "evidence_check": "poisoning_detection", "verified": None},
        ],
        "declared_weaknesses": [{"issue": "Opacus secure_mode disabled", "severity": "medium"}],
        "overall_verdict": verdict,
    }
    path = tmp_path / "compliance_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def _issue(tmp_path, key=KEY, verdict="PASS"):
    report = _report(tmp_path, verdict)
    data = collect(report, key, issued_at="2026-10-18T10:05:00+00:00")
    pdf = tmp_path / "compliance_certificate.pdf"
    render_pdf(data, pdf)
    record = tmp_path / "compliance_certificate.json"
    record.write_text(json.dumps(sidecar(data, pdf)))
    return report, pdf, record, data


def test_collect_reads_checks_and_controls(tmp_path):
    data = collect(_report(tmp_path), KEY)
    assert dict(data.checks) == {
        "Signed audit certificates": "Passed",
        "Privacy budget logs": "Passed",
        "Calibration budget composition": "Failed",
        "Data poisoning detection": "Not run",
    }
    assert (data.controls_verified, data.controls_total) == (2, 4)
    assert data.cert_id == certificate_id(data.digest) and data.cert_id.startswith("R26-RC-")
    assert data.digest in qr_payload(data.digest)


def test_pdf_is_a_real_pdf_and_embeds_the_hash(tmp_path):
    _, pdf, record, data = _issue(tmp_path)
    raw = pdf.read_bytes()
    assert raw.startswith(b"%PDF") and f"sha256:{data.digest}".encode() in raw
    assert json.loads(record.read_text())["hmac_signature"] == data.signature


def test_verify_passes_then_catches_every_kind_of_change(tmp_path):
    report, pdf, record, _ = _issue(tmp_path)
    assert verify_certificate(report, record, pdf, KEY).passed

    assert verify_certificate(report, record, pdf, b"wrong key").signature_valid is False

    original = pdf.read_bytes()
    pdf.write_bytes(original + b"\n% edited")
    assert verify_certificate(report, record, pdf, KEY).pdf_matches is False
    pdf.write_bytes(original)

    report.write_text(report.read_text().replace('"PASS"', '"FAIL"'))
    result = verify_certificate(report, record, pdf, KEY)
    assert not result.report_matches and not result.passed


def test_same_report_gives_the_same_pdf(tmp_path):
    _, pdf, _, _ = _issue(tmp_path)
    first = pdf.read_bytes()
    _, pdf2, _, _ = _issue(tmp_path)
    assert pdf2.read_bytes() == first


def test_unsigned_certificate_says_so(tmp_path):
    _, pdf, record, data = _issue(tmp_path, key=None, verdict="INCOMPLETE")
    assert data.signature is None and json.loads(record.read_text())["algorithm"] is None
    assert verify_certificate(tmp_path / "compliance_report.json", record, pdf, None).passed


def test_command_builds_and_verifies(tmp_path, monkeypatch):
    from scripts import build_certificate

    report = _report(tmp_path)
    monkeypatch.setenv("R26_AUDIT_HMAC_KEY", KEY.decode())
    monkeypatch.setattr(sys, "argv", ["rc-certificate", "--report", str(report)])
    build_certificate.main()
    assert (tmp_path / "compliance_certificate.pdf").is_file()
    monkeypatch.setattr(sys, "argv", ["rc-certificate", "--report", str(report), "--verify"])
    with pytest.raises(SystemExit) as done:
        build_certificate.main()
    assert done.value.code == 0
