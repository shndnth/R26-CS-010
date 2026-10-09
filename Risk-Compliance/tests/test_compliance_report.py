"""Compliance report assembly with evidence from Utility Evaluation."""

from __future__ import annotations

import json
import sys

from scripts import build_compliance_report as report
from scripts.build_compliance_report import CHECK_SOURCES, check_passed


def test_filtered_poisoning_verdict_counts_as_pass():
    assert check_passed({"verdict": "FILTERED"}, "verdict") is True
    assert check_passed({"verdict": "PASS"}, "verdict") is True
    assert check_passed({"verdict": "FAIL"}, "verdict") is False


def write(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload))


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["rc-report", *args])
    report.main()


def all_own_checks(directory):
    write(directory, "certificate_verification.json", {"all_valid": True})
    write(directory, "budget_audit.json", {"all_passed": True})
    write(directory, "reconstruction_attack.json", {"audit_log_reconciliation": {"all_reconcile": True}})
    write(directory, "anonymity.json", {"verdict": "PASS"})
    write(directory, "manifest_verification.json", {"all_valid": True})
    write(directory, "calibration_composition.json", {"verdict": "PASS"})


def test_without_utility_evidence_is_incomplete(tmp_path, monkeypatch):
    own = tmp_path / "own"
    all_own_checks(own)
    out = tmp_path / "report.json"
    run(monkeypatch, "--outputs-dir", str(own), "--output", str(out))
    payload = json.loads(out.read_text())
    assert payload["overall_verdict"] == "INCOMPLETE"
    assert "poisoning_detection" in payload["checks_not_run"]


def test_full_evidence_passes(tmp_path, monkeypatch):
    own, utility = tmp_path / "own", tmp_path / "utility"
    all_own_checks(own)
    write(utility, "poisoning_audit.json", {"verdict": "FILTERED"})
    write(utility, "utility_results.json", {"utility_retention_pct": 42.0, "fid": 90.0,
                                            "experiments_missing": []})
    out = tmp_path / "report.json"
    run(monkeypatch, "--outputs-dir", str(own), "--utility-dir", str(utility), "--output", str(out))
    payload = json.loads(out.read_text())
    assert payload["overall_verdict"] == "PASS"
    assert payload["utility_evidence"]["utility_retention_pct"] == 42.0


def test_every_source_is_located():
    assert {where for where, _, _ in CHECK_SOURCES.values()} <= {"own", "utility"}


def test_composition_failure_fails_the_report(tmp_path, monkeypatch):
    own, utility = tmp_path / "own", tmp_path / "utility"
    all_own_checks(own)
    write(own, "calibration_composition.json", {"verdict": "FAIL"})
    write(utility, "poisoning_audit.json", {"verdict": "PASS"})
    out = tmp_path / "report.json"
    run(monkeypatch, "--outputs-dir", str(own), "--utility-dir", str(utility), "--output", str(out))
    payload = json.loads(out.read_text())
    assert payload["overall_verdict"] == "FAIL"
    failed = [m for m in payload["regulatory_mapping"] if m["verified"] is False]
    assert {m["evidence_check"] for m in failed} == {"calibration_composition"}
