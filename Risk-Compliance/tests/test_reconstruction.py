"""Reconstruction attack on the calibration channel."""

from __future__ import annotations

import json

import pytest

from risk_compliance.reconstruction import run_attack, verify_audit_log

EPSILON = 1.8
SENSITIVITIES = {
    "vehicle_density_per_km": 1.0,
    "avg_vehicle_speed_kmh": 2.0,
}

# Real values from the project, so the test reflects the actual data.
TRUE = {"vehicle_density_per_km": 2.0729, "avg_vehicle_speed_kmh": 20.6102}
PUBLISHED = {"vehicle_density_per_km": 2.5139, "avg_vehicle_speed_kmh": 22.0135}
NOISE = {"vehicle_density_per_km": 0.4410, "avg_vehicle_speed_kmh": 1.4033}


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class TestRunAttack:
    def test_error_is_the_difference(self, tmp_path):
        result = run_attack(
            write(tmp_path, "true.json", TRUE),
            write(tmp_path, "pub.json", PUBLISHED),
            EPSILON, SENSITIVITIES,
        )
        by_field = {f.field: f for f in result.fields}
        assert by_field["vehicle_density_per_km"].absolute_error == pytest.approx(0.4410, abs=1e-4)

    def test_expected_error_is_sensitivity_over_epsilon(self, tmp_path):
        result = run_attack(
            write(tmp_path, "true.json", TRUE),
            write(tmp_path, "pub.json", PUBLISHED),
            EPSILON, SENSITIVITIES,
        )
        by_field = {f.field: f for f in result.fields}
        assert by_field["vehicle_density_per_km"].expected_error == pytest.approx(1.0 / 1.8)
        assert by_field["avg_vehicle_speed_kmh"].expected_error == pytest.approx(2.0 / 1.8)

    def test_calibration_ratio_near_one_is_consistent(self, tmp_path):
        result = run_attack(
            write(tmp_path, "true.json", TRUE),
            write(tmp_path, "pub.json", PUBLISHED),
            EPSILON, SENSITIVITIES,
        )
        assert 0.3 < result.calibration_ratio < 3.0
        assert "consistent with" in result.interpretation()

    def test_under_noising_is_flagged(self, tmp_path):
        barely_noised = {k: v * 1.0001 for k, v in TRUE.items()}
        result = run_attack(
            write(tmp_path, "true.json", TRUE),
            write(tmp_path, "pub.json", barely_noised),
            EPSILON, SENSITIVITIES,
        )
        assert result.calibration_ratio < 0.3
        assert "under-noising" in result.interpretation()

    def test_no_matching_fields_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no matching fields"):
            run_attack(
                write(tmp_path, "true.json", {"unrelated": 1.0}),
                write(tmp_path, "pub.json", {"other": 2.0}),
                EPSILON, SENSITIVITIES,
            )


class TestVerifyAuditLog:
    def _audit_log(self, tmp_path, noise):
        return write(tmp_path, "audit.json", {
            "epsilon_calibration": EPSILON,
            "queries": [{"query": k, "noise_added": v} for k, v in noise.items()],
        })

    def test_correct_log_reconciles(self, tmp_path):
        result = verify_audit_log(
            write(tmp_path, "true.json", TRUE),
            write(tmp_path, "pub.json", PUBLISHED),
            self._audit_log(tmp_path, NOISE),
        )
        assert result["all_reconcile"]
        assert result["fields_checked"] == 2

    def test_falsified_noise_is_detected(self, tmp_path):
        wrong = dict(NOISE)
        wrong["vehicle_density_per_km"] = 0.0001
        result = verify_audit_log(
            write(tmp_path, "true.json", TRUE),
            write(tmp_path, "pub.json", PUBLISHED),
            self._audit_log(tmp_path, wrong),
        )
        assert not result["all_reconcile"]
        assert "do not reconcile" in result["interpretation"]
