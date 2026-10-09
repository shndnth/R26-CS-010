"""Independent certificate verification."""

from __future__ import annotations

import json

from risk_compliance.verification import tamper_test, verify, verify_all
from tests.conftest import TEST_KEY, make_run


class TestVerify:
    def test_untampered_report_passes(self, tmp_path):
        run = make_run(tmp_path, "config_a")
        result = verify(run / "audit_report.json", run / "audit_certificate.json", TEST_KEY)
        assert result.passed
        assert result.verdict == "PASS"

    def test_algorithm_is_reported(self, tmp_path):
        run = make_run(tmp_path, "config_a")
        result = verify(run / "audit_report.json", run / "audit_certificate.json", TEST_KEY)
        assert result.algorithm == "HMAC-SHA256"

    def test_modified_report_fails(self, tmp_path):
        run = make_run(tmp_path, "config_a")
        report = run / "audit_report.json"
        payload = json.loads(report.read_text())
        payload["results"]["final_epsilon"] = 0.01
        report.write_text(json.dumps(payload, indent=2))

        result = verify(report, run / "audit_certificate.json", TEST_KEY)
        assert not result.passed
        assert not result.signature_valid
        assert not result.digest_valid

    def test_whitespace_change_is_detected(self, tmp_path):
        run = make_run(tmp_path, "config_a")
        report = run / "audit_report.json"
        report.write_text(report.read_text() + "\n")
        result = verify(report, run / "audit_certificate.json", TEST_KEY)
        assert not result.passed

    def test_wrong_key_fails(self, tmp_path):
        run = make_run(tmp_path, "config_a")
        result = verify(
            run / "audit_report.json", run / "audit_certificate.json", b"wrong-key"
        )
        assert not result.signature_valid

    def test_wrong_key_still_matches_digest(self, tmp_path):
        """The digest covers content only, so it is key-independent."""
        run = make_run(tmp_path, "config_a")
        result = verify(
            run / "audit_report.json", run / "audit_certificate.json", b"wrong-key"
        )
        assert result.digest_valid


class TestVerifyAll:
    def test_finds_every_report(self, runs_dir):
        results = verify_all(runs_dir, TEST_KEY)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_empty_directory_returns_nothing(self, tmp_path):
        assert verify_all(tmp_path, TEST_KEY) == []


class TestTamperTest:
    def test_detects_falsified_epsilon(self, tmp_path):
        run = make_run(tmp_path, "config_a", final_epsilon=0.9970)
        original, result = tamper_test(
            run / "audit_report.json",
            run / "audit_certificate.json",
            TEST_KEY,
            tmp_path / "workspace",
        )
        assert original == 0.9970
        assert not result.passed

    def test_original_files_are_not_modified(self, tmp_path):
        run = make_run(tmp_path, "config_a")
        before = (run / "audit_report.json").read_text()
        tamper_test(
            run / "audit_report.json",
            run / "audit_certificate.json",
            TEST_KEY,
            tmp_path / "workspace",
        )
        assert (run / "audit_report.json").read_text() == before
