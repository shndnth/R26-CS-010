"""Explicit record written for the non-private control run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from privacy_integration.artifacts import ArtifactStore
from privacy_integration.audit.signing import CertificateSigner
from privacy_integration.evaluation.metrics import compute_metrics
from privacy_integration.training.experiment import (
    ExperimentResult,
    _write_non_private_declaration,
)


@pytest.fixture
def declaration(tmp_path):
    paths = ArtifactStore(root=tmp_path).run(seed=42, tag="no_dp").create()
    metrics = compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]))
    result = ExperimentResult(
        tag="no_dp", seed=42, epsilon=None, final_epsilon=None,
        noise_multiplier=None, val_metrics=metrics, test_metrics=metrics,
    )
    _write_non_private_declaration(paths, result, {"seed": 42})
    return paths


class TestNonPrivateDeclaration:
    def test_writes_an_audit_report(self, declaration):
        assert declaration.audit_report.is_file()

    def test_writes_a_signed_certificate(self, declaration):
        assert declaration.audit_certificate.is_file()

    def test_states_dp_was_not_applied(self, declaration):
        payload = json.loads(declaration.audit_report.read_text())
        assert payload["differential_privacy_applied"] is False

    def test_epsilon_is_null_not_zero(self, declaration):
        payload = json.loads(declaration.audit_report.read_text())
        assert payload["privacy_config"]["epsilon"] is None

    def test_warns_against_reading_absence_as_zero(self, declaration):
        payload = json.loads(declaration.audit_report.read_text())
        assert "MUST NOT be read as" in payload["warning"]

    def test_declaration_signature_verifies(self, declaration):
        signer = CertificateSigner(key=b"test-hmac-key-not-for-production")
        # Re-sign with a known key so verification is independent of the env value
        signer.sign(declaration.audit_report, declaration.audit_certificate)
        assert signer.verify(declaration.audit_report, declaration.audit_certificate)

    def test_tampering_with_declaration_is_detected(self, declaration):
        signer = CertificateSigner(key=b"test-hmac-key-not-for-production")
        signer.sign(declaration.audit_report, declaration.audit_certificate)

        payload = json.loads(declaration.audit_report.read_text())
        payload["differential_privacy_applied"] = True   # falsify the claim
        declaration.audit_report.write_text(json.dumps(payload, indent=2))

        assert not signer.verify(declaration.audit_report, declaration.audit_certificate)
