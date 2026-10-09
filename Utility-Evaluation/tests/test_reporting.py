"""Results aggregation."""

from __future__ import annotations

import json

import pytest

from utility_evaluation.reporting import (
    REAL_TO_REAL,
    REAL_TO_SYNTH,
    SYNTH_TO_REAL,
    SYNTH_TO_SYNTH,
    load,
)


def evaluation(label, map50, map50_95, per_class=None):
    return {
        "label": label,
        "map50": map50,
        "map50_95": map50_95,
        "per_class": per_class or {},
    }


def write_evaluations(tmp_path, records):
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "evaluations.json").write_text(json.dumps(records), encoding="utf-8")
    return outputs


class TestRetention:
    def test_retention_is_transfer_over_baseline(self, tmp_path):
        outputs = write_evaluations(tmp_path, [
            evaluation(SYNTH_TO_REAL, 0.35, 0.20),
            evaluation(REAL_TO_REAL, 0.70, 0.40),
        ])
        report = load(outputs)
        assert report.retention_pct == pytest.approx(50.0)

    def test_retention_is_none_without_the_baseline(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_REAL, 0.35, 0.20)])
        assert load(outputs).retention_pct is None

    def test_headline_explains_what_is_missing(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_SYNTH, 0.5, 0.3)])
        assert "cannot be computed" in load(outputs).headline()

    def test_headline_states_the_percentage(self, tmp_path):
        outputs = write_evaluations(tmp_path, [
            evaluation(SYNTH_TO_REAL, 0.35, 0.20),
            evaluation(REAL_TO_REAL, 0.70, 0.40),
        ])
        assert "50%" in load(outputs).headline()


class TestSimToRealGap:
    def test_gap_is_the_drop_on_real_data(self, tmp_path):
        outputs = write_evaluations(tmp_path, [
            evaluation(SYNTH_TO_SYNTH, 0.60, 0.35),
            evaluation(SYNTH_TO_REAL, 0.35, 0.20),
        ])
        assert load(outputs).sim_to_real_gap == pytest.approx(0.15)

    def test_gap_is_none_when_incomplete(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_SYNTH, 0.60, 0.35)])
        assert load(outputs).sim_to_real_gap is None


class TestPerClass:
    def test_identifies_the_weakest_class(self, tmp_path):
        outputs = write_evaluations(tmp_path, [
            evaluation(SYNTH_TO_SYNTH, 0.6, 0.35, {
                "car": {"ap50": 0.80, "ap50_95": 0.55},
                "pedestrian": {"ap50": 0.30, "ap50_95": 0.15},
                "cyclist": {"ap50": 0.40, "ap50_95": 0.22},
            }),
        ])
        name, value = load(outputs).weakest_class()
        assert name == "pedestrian"
        assert value == pytest.approx(0.15)

    def test_weakest_class_is_none_without_per_class_data(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_SYNTH, 0.6, 0.35)])
        assert load(outputs).weakest_class() is None


class TestAggregation:
    def test_missing_experiments_are_listed(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_SYNTH, 0.6, 0.35)])
        missing = load(outputs).as_dict()["experiments_missing"]
        assert SYNTH_TO_REAL in missing
        assert REAL_TO_REAL in missing
        assert REAL_TO_SYNTH in missing

    def test_picks_up_fid_when_present(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_SYNTH, 0.6, 0.35)])
        (outputs / "fid.json").write_text(
            json.dumps({"fid": 42.5, "n_real": 1000, "n_synthetic": 1200}), encoding="utf-8"
        )
        report = load(outputs)
        assert report.fid == 42.5
        assert report.fid_sample_size == 1000

    def test_picks_up_poisoning_verdict(self, tmp_path):
        outputs = write_evaluations(tmp_path, [evaluation(SYNTH_TO_SYNTH, 0.6, 0.35)])
        (outputs / "poisoning_audit.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )
        assert load(outputs).poisoning_verdict == "PASS"

    def test_empty_outputs_yields_no_experiments(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        assert load(outputs).experiments == {}
