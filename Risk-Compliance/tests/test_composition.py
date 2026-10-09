"""Calibration budget composition."""

from __future__ import annotations

import json
import sys

import pytest

from risk_compliance.composition import analyse
from scripts import calibration_composition

SENSITIVITIES = {
    "vehicle_density_per_km": 1.0, "pedestrian_density_per_km": 1.0, "avg_vehicle_speed_kmh": 2.0,
    "avg_pedestrian_speed_kmh": 1.0, "intersection_count_per_km": 1.0,
    "weather_distribution": 1.0, "time_of_day_distribution": 1.0,
}
STATED = 1.8


def legacy_log(per_query_epsilon: float = STATED) -> dict:
    """The log as the calibration code first wrote it: epsilon_used holds the scale."""
    return {"epsilon_calibration": STATED, "queries": [
        {"query": name, "sensitivity": s, "noise_added": 0.1, "epsilon_used": round(s / per_query_epsilon, 6)}
        for name, s in SENSITIVITIES.items()
    ]}


def explicit_log(per_query_epsilon: float) -> dict:
    return {"epsilon_calibration": STATED, "queries": [
        {"query": name, "sensitivity": s, "noise_added": 0.1, "epsilon_used": s / per_query_epsilon,
         "laplace_scale": s / per_query_epsilon, "epsilon_spent": per_query_epsilon}
        for name, s in SENSITIVITIES.items()
    ]}


def test_full_budget_per_query_composes_to_seven_times_the_claim():
    result = analyse(legacy_log(), {"calibration_epsilon_spent": STATED})
    assert result.field_meaning.startswith("Laplace scale")
    assert result.composed_total == pytest.approx(7 * STATED)
    assert result.verdict == "FAIL"
    assert result.per_query_budget_for_claim == pytest.approx(STATED / 7)
    assert "12.6" in result.interpretation()


def test_explicit_fields_from_the_current_calibration_code():
    result = analyse(explicit_log(STATED), {"calibration_epsilon_spent": STATED})
    assert result.field_meaning.startswith("explicit")
    assert result.verdict == "FAIL"


def test_budget_split_across_queries_passes():
    result = analyse(explicit_log(STATED / 7), {"calibration_epsilon_spent": STATED})
    assert result.composed_total == pytest.approx(STATED)
    assert result.verdict == "PASS"


def test_legacy_field_holding_per_query_epsilon():
    log = {"epsilon_calibration": STATED, "queries": [
        {"query": n, "sensitivity": s, "noise_added": 0.1, "epsilon_used": STATED / 7}
        for n, s in SENSITIVITIES.items()
    ]}
    result = analyse(log, {"calibration_epsilon_spent": STATED})
    assert result.field_meaning == "per-query epsilon"
    assert result.verdict == "PASS"


def test_empty_log_is_rejected():
    with pytest.raises(ValueError, match="no queries"):
        analyse({"epsilon_calibration": STATED, "queries": []})


def test_command_writes_the_result(tmp_path, monkeypatch):
    log = tmp_path / "calibration_audit_log.json"
    log.write_text(json.dumps(legacy_log()))
    out = tmp_path / "calibration_composition.json"
    monkeypatch.setattr(sys, "argv", ["rc", "--audit-log", str(log), "--output", str(out)])
    calibration_composition.main()
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "FAIL"
    assert payload["composed_total_epsilon"] == pytest.approx(12.6)
