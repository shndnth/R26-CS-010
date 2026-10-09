"""Metadata leakage scanning."""

from __future__ import annotations

from risk_compliance.leakage import scan

SUFFIXES = frozenset({".json", ".csv", ".txt", ".md"})


class TestScan:
    def test_clean_directory_passes(self, tmp_path):
        (tmp_path / "clean.json").write_text('{"value": 1}', encoding="utf-8")
        result = scan(tmp_path, SUFFIXES)
        assert result.passed
        assert result.verdict == "PASS"

    def test_counts_files_scanned(self, tmp_path):
        for i in range(3):
            (tmp_path / f"f{i}.json").write_text("{}", encoding="utf-8")
        assert scan(tmp_path, SUFFIXES).files_scanned == 3

    def test_ignores_unlisted_suffixes(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"binary")
        assert scan(tmp_path, SUFFIXES).files_scanned == 0

    def test_detects_absolute_unix_path(self, tmp_path):
        (tmp_path / "a.json").write_text('{"p": "/home/someuser/project"}', encoding="utf-8")
        result = scan(tmp_path, SUFFIXES)
        assert "absolute_unix_path" in result.by_issue()

    def test_detects_email(self, tmp_path):
        (tmp_path / "a.txt").write_text("contact person@example.com", encoding="utf-8")
        assert "email_address" in scan(tmp_path, SUFFIXES).by_issue()

    def test_detects_ip_address(self, tmp_path):
        (tmp_path / "a.txt").write_text("host 192.168.1.42", encoding="utf-8")
        assert "ip_address" in scan(tmp_path, SUFFIXES).by_issue()

    def test_detects_gps_coordinate(self, tmp_path):
        (tmp_path / "a.txt").write_text("at 6.927079, 79.861244", encoding="utf-8")
        assert "gps_coordinate" in scan(tmp_path, SUFFIXES).by_issue()

    def test_detects_possible_credential(self, tmp_path):
        (tmp_path / "a.txt").write_text("api_key = abc123def", encoding="utf-8")
        assert "possible_credential" in scan(tmp_path, SUFFIXES).by_issue()

    def test_verdict_requires_review_when_findings_exist(self, tmp_path):
        (tmp_path / "a.txt").write_text("password: hunter2", encoding="utf-8")
        assert scan(tmp_path, SUFFIXES).verdict == "REVIEW REQUIRED"

    def test_recurses_into_subdirectories(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "deep.txt").write_text("token: xyz", encoding="utf-8")
        assert scan(tmp_path, SUFFIXES).findings


class TestPatternPrecision:
    def test_json_number_list_is_not_a_coordinate(self, tmp_path):
        from risk_compliance.leakage import scan

        (tmp_path / "a.json").write_text('{"values": [\n  0.5303,\n  0.4199\n]}')
        assert scan(tmp_path, frozenset({".json"})).passed

    def test_coordinate_on_one_line_is_found(self, tmp_path):
        from risk_compliance.leakage import scan

        (tmp_path / "a.json").write_text('{"where": "6.9271, 79.8612"}')
        assert not scan(tmp_path, frozenset({".json"})).passed

    def test_temporary_and_mounted_paths_are_found(self, tmp_path):
        from risk_compliance.leakage import scan

        (tmp_path / "a.json").write_text('{"target": "/tmp/run/privacy/runs/seed_42"}')
        assert not scan(tmp_path, frozenset({".json"})).passed
