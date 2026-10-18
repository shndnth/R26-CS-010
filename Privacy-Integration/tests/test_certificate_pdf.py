"""The printable privacy certificate must match privacy_certificate.json and expose any change."""

from __future__ import annotations

import json

from privacy_integration.audit.certificate_pdf import collect, record, render, verify_pdf
from privacy_integration.audit.signing import CertificateSigner

KEY = b"test-hmac-key-not-for-production"


def _certificate(tmp_path):
    cert = {
        "configurations": {
            "no_dp": {
                "target_epsilon": None,
                "replications": 3,
                "achieved_epsilon": {"mean": None},
                "utility_auc": {"mean": 0.829},
                "attack_auc": {"mean": 0.639},
            },
            "config_a": {
                "target_epsilon": 1.0,
                "replications": 3,
                "achieved_epsilon": {"mean": 0.99},
                "utility_auc": {"mean": 0.621},
                "attack_auc": {"mean": 0.504},
            },
        },
        "privacy_mechanism": {
            "calibration_stage": {"queries": 7, "epsilon_total": 1.8, "epsilon_per_query": 0.2571},
            "training_stage": {"clipping_norm_C": 1.0, "delta": 1e-5, "epsilon_ceiling": 10.0},
        },
        "stress_test": {"status": "not run"},
        "calibrated_attack": {"status": "not run"},
        "declared_limitations": [{"issue": "Opacus secure_mode disabled", "severity": "medium"}],
        "certification": {
            "privacy_enforcement": "PASS",
            "empirical_validation": "PASS",
            "overall_privacy_verdict": "PASS",
        },
    }
    path = tmp_path / "privacy_certificate.json"
    path.write_text(json.dumps(cert, indent=2))
    sig = tmp_path / "privacy_certificate_signature.json"
    CertificateSigner(key=KEY).sign(path, sig)
    return path, sig


def _issue(tmp_path):
    cert, sig = _certificate(tmp_path)
    data = collect(cert, sig, issued_at="2026-10-18T10:00:00+00:00")
    pdf = tmp_path / "privacy_certificate.pdf"
    render(data, pdf)
    rec = tmp_path / "privacy_certificate_pdf.json"
    rec.write_text(json.dumps(record(data, pdf)))
    return cert, pdf, rec, data


def test_rows_signature_and_control_first(tmp_path):
    _, _, _, data = _issue(tmp_path)
    assert data.rows[0][0].startswith("No DP") and data.rows[1][1] == "1"
    assert data.signature and data.cert_id.startswith("R26-PI-")
    assert any("1.8 in total" in line for line in data.lines)


def test_pdf_verifies_then_catches_changes(tmp_path):
    cert, pdf, rec, data = _issue(tmp_path)
    assert pdf.read_bytes().startswith(b"%PDF")
    assert verify_pdf(cert, pdf, rec).passed

    original = pdf.read_bytes()
    pdf.write_bytes(original + b"\n% edited")
    assert not verify_pdf(cert, pdf, rec).pdf_matches
    pdf.write_bytes(original)

    cert.write_text(cert.read_text().replace('"PASS"', '"FAIL"', 1))
    check = verify_pdf(cert, pdf, rec)
    assert not check.certificate_matches and not check.pdf_embeds_hash


def test_stale_signature_is_not_printed(tmp_path):
    cert, sig = _certificate(tmp_path)
    cert.write_text(cert.read_text() + " ")
    assert collect(cert, sig).signature is None


def test_same_input_gives_same_pdf(tmp_path):
    _, pdf, _, _ = _issue(tmp_path)
    first = pdf.read_bytes()
    _, pdf2, _, _ = _issue(tmp_path)
    assert pdf2.read_bytes() == first


def test_rounding_matches_the_reports():
    from privacy_integration.audit.certificate_pdf import _fmt

    assert _fmt(0.6385) == "0.639" and _fmt(0.6545) == "0.655" and _fmt(None) == "n/a"
    assert _fmt(0.99, 2) == "0.99"
