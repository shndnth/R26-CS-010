"""Utility Evaluation handoffs: utility_results.json."""

from __future__ import annotations

import json

import pytest

from r26_contracts import ContractError, validate

UTILITY = {
    "experiments": {
        "synthetic -> real": {"map50": 0.4, "map50_95": 0.2,
                              "per_class": {"car": {"ap50": 0.5, "ap50_95": 0.3}, "cyclist": None}},
    },
    "utility_retention_pct": 44.0,
    "sim_to_real_gap_map50_95": 0.2,
    "fid": 91.3,
    "fid_sample_size": 1000,
    "poisoning_verdict": "FILTERED",
    "experiments_missing": [],
}


def test_valid_utility_results_pass():
    validate(UTILITY, "utility_results")


def test_unknown_experiment_label_is_rejected():
    broken = {**UTILITY, "experiments": {"synthetic->real": UTILITY["experiments"]["synthetic -> real"]}}
    with pytest.raises(ContractError, match="experiments"):
        validate(broken, "utility_results")


def test_missing_field_is_reported_by_name():
    broken = {k: v for k, v in UTILITY.items() if k != "fid"}
    with pytest.raises(ContractError) as info:
        validate(broken, "utility_results")
    assert any("'fid'" in p for p in info.value.problems)


def test_map_out_of_range_is_rejected():
    broken = json.loads(json.dumps(UTILITY))
    broken["experiments"]["synthetic -> real"]["map50"] = 41.0
    with pytest.raises(ContractError):
        validate(broken, "utility_results")
