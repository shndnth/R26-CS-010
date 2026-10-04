"""The execution engine, driven by stand-in component scripts."""

from __future__ import annotations

import json

import pytest

from r26_pipeline.runner import Pipeline
from r26_pipeline.stages import Stage

VALID_SIGNATURE = {
    "report_file": "audit_report.json", "sha256_content": "a" * 64, "hmac_signature": "b" * 64,
    "algorithm": "HMAC-SHA256", "signed_at": "2026-10-04T00:00:00Z",
}


def writer(name, component, target, payload, **kwargs):
    return Stage(name, component, f"write {target.name}", module="scripts.write_json",
                 args=(str(target), json.dumps(payload)), outputs=(target,), **kwargs)


@pytest.fixture
def ws(config):
    return config.workspace


def make(config, stages):
    return Pipeline(config, stages=stages, echo=False)


def test_runs_in_order_and_records_lineage(config, ws):
    a = writer("a", "privacy", ws / "a.json", {"x": 1})
    b = writer("b", "utility", ws / "b.json", {"y": 2}, inputs=(ws / "a.json",), needs=("a",))
    pipeline = make(config, [a, b])
    assert pipeline.run([a, b]) == 0
    record = pipeline.state.get("b")
    assert record.status == "done" and record.code
    assert (ws / "logs" / "b.log").read_text().count("wrote b.json") == 1
    assert str(ws / "b.json") in record.outputs


def test_second_run_is_up_to_date(config, ws):
    a = writer("a", "privacy", ws / "a.json", {"x": 1})
    make(config, [a]).run([a])
    pipeline = make(config, [a])
    assert pipeline.status(a).label == "done"


def test_changed_input_makes_downstream_stale(config, ws):
    source = ws / "input.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    a = writer("a", "privacy", ws / "a.json", {"x": 1}, inputs=(source,))
    make(config, [a]).run([a])
    source.write_text('{"changed": true}')
    assert make(config, [a]).status(a).label == "stale"


def test_edited_output_is_detected(config, ws):
    a = writer("a", "privacy", ws / "a.json", {"x": 1})
    make(config, [a]).run([a])
    (ws / "a.json").write_text('{"x": 999}')
    status = make(config, [a]).status(a)
    assert status.label == "stale" and "a.json" in status.detail


def test_failure_stops_the_run(config, ws):
    bad = Stage("bad", "utility", "fails", module="scripts.fail")
    after = writer("after", "utility", ws / "after.json", {})
    pipeline = make(config, [bad, after])
    assert pipeline.run([bad, after]) == 1
    assert pipeline.state.get("bad").status == "failed"
    assert "code 3" in pipeline.state.get("bad").detail
    assert pipeline.state.get("after") is None


def test_contract_violation_fails_the_stage(config, ws):
    target = ws / "audit_certificate.json"
    stage = writer("v", "privacy", target, {**VALID_SIGNATURE, "sha256_content": "not a digest"},
                   contracts=((target, "audit_certificate"),))
    pipeline = make(config, [stage])
    assert pipeline.run([stage]) == 1
    assert "audit_certificate" in pipeline.state.get("v").detail


def test_valid_contract_passes(config, ws):
    target = ws / "audit_certificate.json"
    stage = writer("v", "privacy", target, VALID_SIGNATURE, contracts=((target, "audit_certificate"),))
    assert make(config, [stage]).run([stage]) == 0


def test_gate_stops_the_run(config, ws):
    stage = writer("g", "compliance", ws / "g.json", {}, gate=lambda _ws: "integrity broken")
    pipeline = make(config, [stage])
    assert pipeline.run([stage]) == 1
    assert "integrity broken" in pipeline.state.get("g").detail


def test_needs_must_have_completed(config, ws):
    a = writer("a", "privacy", ws / "a.json", {})
    b = writer("b", "privacy", ws / "b.json", {}, needs=("a",))
    pipeline = make(config, [a, b])
    assert pipeline.run([b]) == 1
    assert "needs a" in pipeline.state.get("b").detail


def test_missing_declared_output_fails(config, ws):
    stage = Stage("s", "privacy", "writes the wrong file", module="scripts.write_json",
                  args=(str(ws / "other.json"), "{}"), outputs=(ws / "expected.json",))
    pipeline = make(config, [stage])
    assert pipeline.run([stage]) == 1
    assert "expected.json" in pipeline.state.get("s").detail


def test_skipped_stage_cascades(config, ws):
    a = Stage("a", "privacy", "optional", module="scripts.write_json", skip_reason="input not set")
    b = writer("b", "privacy", ws / "b.json", {}, needs=("a",))
    pipeline = make(config, [a, b])
    assert pipeline.run([a, b]) == 0
    assert pipeline.state.get("b").status == "skipped"
    assert "needs a" in pipeline.state.get("b").detail


def test_dry_run_executes_nothing(config, ws):
    a = writer("a", "privacy", ws / "a.json", {})
    pipeline = make(config, [a])
    assert pipeline.run([a], dry_run=True) == 0
    assert not (ws / "a.json").exists() and pipeline.state.get("a") is None


def test_forecast_marks_downstream_of_a_rerun(config, ws):
    source = ws / "input.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    a = writer("a", "privacy", ws / "a.json", {}, inputs=(source,))
    b = writer("b", "utility", ws / "b.json", {}, inputs=(ws / "a.json",), needs=("a",))
    make(config, [a, b]).run([a, b])
    source.write_text("[]")
    forecast = make(config, [a, b]).forecast([a, b])
    assert forecast["a"].label == "stale" and forecast["b"].label == "stale"
    assert "upstream" in forecast["b"].detail


def test_component_code_change_makes_its_stages_stale(config, ws, fake_repo):
    a = writer("a", "privacy", ws / "a.json", {})
    make(config, [a]).run([a])
    (fake_repo / "Privacy-Integration" / "scripts" / "helper.py").write_text("VALUE = 1\n")
    assert make(config, [a]).status(a).label == "stale"


def test_clean_paths_are_removed_first(config, ws):
    leftover = ws / "old" / "stale.txt"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("x")
    a = writer("a", "privacy", ws / "a.json", {}, clean=(ws / "old",))
    make(config, [a]).run([a])
    assert not leftover.exists()
