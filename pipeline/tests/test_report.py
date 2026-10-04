"""Integrated verdict and release assembly."""

from __future__ import annotations

import json

from r26_pipeline.report import assemble_release, overall_verdict
from r26_pipeline.stages import RunWorkspace


def test_verdict_rules():
    assert overall_verdict("PASS", "COMPLETE", "PASS", []) == "PASS"
    assert overall_verdict("FAIL", "COMPLETE", "PASS", []) == "FAIL"
    assert overall_verdict("PASS", "COMPLETE", "FAIL", []) == "FAIL"
    assert overall_verdict("PASS", "INCOMPLETE", "PASS", []) == "INCOMPLETE"
    assert overall_verdict("PASS", "COMPLETE", "INCOMPLETE", []) == "INCOMPLETE"
    assert overall_verdict("PASS", "COMPLETE", "PASS", ["utility.fid"]) == "INCOMPLETE"


def test_release_copies_deliverables_and_leaves_out_the_audit_log(tmp_path):
    ws = RunWorkspace(tmp_path)
    (ws.calibration).mkdir(parents=True)
    (ws.calibration / "calibration_stats.json").write_text("{}")
    (ws.calibration / "calibration_audit_log.json").write_text('{"queries": []}')
    ws.utility_results.mkdir(parents=True)
    (ws.utility_results / "utility_results.json").write_text("{}")

    copied = assemble_release(ws)
    assert "calibration_stats.json" in copied and "utility_results.json" in copied
    assert not (ws.release / "calibration_audit_log.json").exists()
    manifest = json.loads((ws.release / "MANIFEST.json").read_text())
    assert set(manifest) == {"calibration_stats.json", "utility_results.json"}
