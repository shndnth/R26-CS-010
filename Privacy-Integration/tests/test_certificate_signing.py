import json
import tempfile
import unittest
from pathlib import Path

from privacy_integration.audit.signing import CertificateSigner


class TestCertificateSigner(unittest.TestCase):

    def setUp(self):
        self.signer = CertificateSigner(key=b"test-key")
        self.tmpdir = Path(tempfile.mkdtemp())
        self.report_path = self.tmpdir / "test_audit_report.json"
        self.cert_path   = self.tmpdir / "test_certificate.json"
        self.report_path.write_text(
            json.dumps({"final_epsilon": 2.99, "budget_respected": True}, indent=2)
        )

    def test_sign_produces_certificate_file(self):
        self.signer.sign(self.report_path, self.cert_path)
        self.assertTrue(self.cert_path.exists())

    def test_certificate_contains_required_fields(self):
        cert = self.signer.sign(self.report_path, self.cert_path)
        for field in ["hmac_signature", "sha256_content", "signed_at", "algorithm"]:
            self.assertIn(field, cert)

    def test_algorithm_is_hmac_sha256(self):
        cert = self.signer.sign(self.report_path, self.cert_path)
        self.assertEqual(cert["algorithm"], "HMAC-SHA256")

    def test_verification_passes_on_unmodified_report(self):
        self.signer.sign(self.report_path, self.cert_path)
        result = self.signer.verify(self.report_path, self.cert_path)
        self.assertTrue(result)

    def test_verification_fails_on_tampered_report(self):
        self.signer.sign(self.report_path, self.cert_path)
        self.report_path.write_text(
            json.dumps({"final_epsilon": 0.1, "budget_respected": False}, indent=2)
        )
        result = self.signer.verify(self.report_path, self.cert_path)
        self.assertFalse(result)

    def test_signature_is_deterministic_for_same_content(self):
        cert1 = self.signer.sign(self.report_path, self.cert_path)
        cert2 = self.signer.sign(self.report_path, self.cert_path)
        self.assertEqual(cert1["hmac_signature"], cert2["hmac_signature"])

    def test_different_content_produces_different_signature(self):
        cert1 = self.signer.sign(self.report_path, self.cert_path)
        self.report_path.write_text(json.dumps({"final_epsilon": 9.9}))
        cert2 = self.signer.sign(self.report_path, self.cert_path)
        self.assertNotEqual(cert1["hmac_signature"], cert2["hmac_signature"])


if __name__ == "__main__":
    unittest.main()
