"""Privacy Integration handoffs: per-run audit reports."""

from __future__ import annotations

import pytest

from r26_contracts import ContractError, validate

DP_REPORT = {
    "project": "R26-CS-010", "member": "IT22309556", "generated_at": "2026-10-04T00:00:00Z",
    "privacy_config": {"noise_multiplier_sigma": 0.97},
    "results": {"final_epsilon": 0.99, "total_steps": 3000, "budget_respected": True},
}

NO_DP = {
    "differential_privacy_applied": False, "config": "no_dp", "seed": 42,
    "privacy_config": {"epsilon": None, "delta": None},
}


def test_audit_report_accepts_dp_and_declaration_forms():
    validate(DP_REPORT, "audit_report")
    validate(NO_DP, "audit_report")


def test_declaration_with_an_epsilon_is_rejected():
    broken = {**NO_DP, "privacy_config": {"epsilon": 0.0}}
    with pytest.raises(ContractError):
        validate(broken, "audit_report")
