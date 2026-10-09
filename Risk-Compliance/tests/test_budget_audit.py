"""Privacy budget audit."""

from __future__ import annotations

import json

from risk_compliance.budget_audit import audit_all, audit_run
from tests.conftest import make_run

CEILING = 10.0
TOLERANCE = 1e-4


class TestPrivateRuns:
    def test_consistent_run_passes(self, tmp_path):
        run = make_run(tmp_path, "config_b", final_epsilon=2.9988, steps=100)
        result = audit_run(run, CEILING, TOLERANCE)
        assert result.passed
        assert result.steps_logged == 100

    def test_reported_matches_logged(self, tmp_path):
        run = make_run(tmp_path, "config_b", final_epsilon=2.9988, steps=100)
        result = audit_run(run, CEILING, TOLERANCE)
        assert result.agreement
        assert abs(result.reported_final_epsilon - result.logged_final_epsilon) < TOLERANCE

    def test_epsilon_is_monotonic(self, tmp_path):
        run = make_run(tmp_path, "config_b", steps=200)
        assert audit_run(run, CEILING, TOLERANCE).monotonic

    def test_ceiling_breach_is_detected(self, tmp_path):
        run = make_run(tmp_path, "config_x", final_epsilon=2.9988, ceiling_breach=True)
        result = audit_run(run, CEILING, TOLERANCE)
        assert not result.ceiling_respected
        assert result.verdict == "FAIL"

    def test_mismatch_between_report_and_log_is_detected(self, tmp_path):
        run = make_run(tmp_path, "config_b", final_epsilon=2.9988, steps=100)
        report_path = run / "audit_report.json"
        payload = json.loads(report_path.read_text())
        payload["results"]["final_epsilon"] = 0.5     # contradicts the log
        report_path.write_text(json.dumps(payload, indent=2))

        result = audit_run(run, CEILING, TOLERANCE)
        assert not result.agreement
        assert result.verdict == "FAIL"

    def test_missing_budget_log_fails_for_a_private_run(self, tmp_path):
        run = make_run(tmp_path, "config_b")
        (run / "budget_log.csv").unlink()
        result = audit_run(run, CEILING, TOLERANCE)
        assert not result.passed
        assert any("missing" in note for note in result.notes)

    def test_slight_target_overshoot_is_noted_not_failed(self, tmp_path):
        run = make_run(tmp_path, "config_b", final_epsilon=3.0046, steps=100)
        result = audit_run(run, CEILING, TOLERANCE)
        assert result.passed
        assert any("RDP accounting granularity" in note for note in result.notes)


class TestNonPrivateBaseline:
    def test_baseline_passes_without_a_budget_log(self, tmp_path):
        run = make_run(tmp_path, "no_dp", private=False)
        result = audit_run(run, CEILING, TOLERANCE)
        assert result.passed

    def test_baseline_epsilon_is_none_not_zero(self, tmp_path):
        run = make_run(tmp_path, "no_dp", private=False)
        result = audit_run(run, CEILING, TOLERANCE)
        assert result.reported_final_epsilon is None

    def test_baseline_is_explained_in_notes(self, tmp_path):
        run = make_run(tmp_path, "no_dp", private=False)
        result = audit_run(run, CEILING, TOLERANCE)
        assert any("non-private control" in note for note in result.notes)


class TestAuditAll:
    def test_audits_every_run(self, runs_dir):
        results = audit_all(runs_dir, CEILING, TOLERANCE)
        assert len(results) == 2
        assert all(r.passed for r in results)
