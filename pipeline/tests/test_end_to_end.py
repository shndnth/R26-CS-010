"""The real pipeline on fixture data, every component in the current environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import fixture
import pytest

from r26_pipeline.cli import main

pytestmark = pytest.mark.skipif(os.environ.get("R26_E2E") != "1", reason="set R26_E2E=1 to run")

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    fixture.build_dataset(root)
    fixture.build_kitti(root)
    fixture.build_calibration_source(root)
    (root / "pipeline.yaml").write_text(f"""
workspace: ./run
repository: {REPOSITORY.as_posix()}
inputs:
  encrypted_dataset: ./output
  calibration_source: ./calibration_source/calibration_stats_source_TRUE_VALUES.json
  kitti: ./kitti_raw/training
privacy:
  seeds: [42]
  configs: [no_dp, config_a]
  epochs: 1
  shadow_epochs: 1
  attack_epochs: 3
  device: cpu
utility:
  epochs: 2
  imgsz: 160
  workers: 0
  device: cpu
  fid_limit: 20
""")
    os.environ.setdefault("R26_DATASET_AES_KEY", fixture.KEY.decode())
    os.environ.setdefault("R26_AUDIT_HMAC_KEY", fixture.HMAC_KEY)
    assert main(["--config", str(root / "pipeline.yaml"), "run"]) == 0
    return root


def test_every_stage_completed(run_dir):
    state = json.loads((run_dir / "run" / "pipeline_state.json").read_text())
    assert {r["status"] for r in state["stages"].values()} == {"done"}


def test_integrated_verdict(run_dir):
    report = json.loads((run_dir / "run" / "release" / "pipeline_report.json").read_text())
    assert report["privacy"]["verdict"] == "PASS"
    assert report["utility"]["status"] == "COMPLETE"
    assert report["compliance"]["release_leakage_scan"] == "PASS"
    assert report["compliance"]["checks_failed"] == []
    assert report["compliance"]["checks_not_run"] == []
    assert report["overall_verdict"] == "PASS"


def test_calibration_reports_the_sequential_total(run_dir):
    stats = json.loads((run_dir / "run" / "release" / "calibration_stats.json").read_text())
    assert stats["calibration_epsilon_spent"] == 7 * stats["calibration_epsilon_per_query"]
    composition = json.loads((run_dir / "run" / "compliance" / "calibration_composition.json").read_text())
    assert composition["verdict"] == "PASS"


def test_release_holds_no_local_paths_or_audit_log(run_dir):
    release = run_dir / "run" / "release"
    assert not (release / "calibration_audit_log.json").exists()
    for path in release.glob("*.json"):
        assert str(run_dir) not in path.read_text(), path.name


def test_handoffs_validate(run_dir):
    assert main(["--config", str(run_dir / "pipeline.yaml"), "validate"]) == 0


def test_second_run_does_nothing(run_dir, capsys):
    capsys.readouterr()
    assert main(["--config", str(run_dir / "pipeline.yaml"), "run"]) == 0
    out = capsys.readouterr().out
    assert out.count("up to date") == out.count("[")


def test_tampered_dataset_stops_at_the_manifest_gate(run_dir):
    frame = run_dir / "run" / "decrypted" / "town01" / "rgb_f000000.png"
    frame.write_bytes(b"tampered")
    code = main(["--config", str(run_dir / "pipeline.yaml"), "run", "--only", "compliance.verify_manifest"])
    state = json.loads((run_dir / "run" / "pipeline_state.json").read_text())
    assert code == 1
    assert "gate failed" in state["stages"]["compliance.verify_manifest"]["detail"]
