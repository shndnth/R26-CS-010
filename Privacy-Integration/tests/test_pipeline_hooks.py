"""Hooks used by the integrated pipeline: outputs location, label gate, utility section."""

from __future__ import annotations

import json
import shutil

from PIL import Image

from privacy_integration.artifacts import ArtifactStore
from privacy_integration.data.dataset import CarlaFrameDataset
from privacy_integration.paths import outputs_root
from privacy_integration.settings import settings
from scripts.build_certificate import utility_section

POSITIVE = "1 0.5 0.5 0.1 0.2"


def make_town(root, frames=6):
    town = root / "town01"
    (town / "labels").mkdir(parents=True)
    for i in range(frames):
        Image.new("RGB", (80, 60)).save(town / f"rgb_f{i:06d}.png")
        (town / "labels" / f"bbox_f{i:06d}.txt").write_text(POSITIVE if i % 2 else "")
    return town


class TestOutputsRoot:
    def test_defaults_to_component_outputs(self, monkeypatch):
        monkeypatch.delenv("R26_OUTPUTS_DIR", raising=False)
        assert outputs_root().name == "outputs"

    def test_environment_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("R26_OUTPUTS_DIR", str(tmp_path))
        assert outputs_root() == tmp_path.resolve()
        assert ArtifactStore().root == tmp_path.resolve() / "runs"


class TestLabelGate:
    def test_default_reads_town_labels(self, tmp_path):
        town = make_town(tmp_path)
        dataset = CarlaFrameDataset([town], settings().dataset)
        assert len(dataset) == 6
        assert dataset.stats.gated == 0

    def test_gate_skips_rejected_frames_instead_of_labelling_them_negative(self, tmp_path):
        town = make_town(tmp_path)
        clean = tmp_path / "clean" / "town01"
        shutil.copytree(town / "labels", clean)
        (clean / "bbox_f000001.txt").unlink()
        (clean / "bbox_f000002.txt").unlink()

        dataset = CarlaFrameDataset([town], settings().dataset, labels_root=tmp_path / "clean")
        names = {p.name for p in dataset.paths}
        assert dataset.stats.gated == 2
        assert len(dataset) == 4
        assert "rgb_f000001.png" not in names and "rgb_f000002.png" not in names


class TestUtilitySection:
    def test_complete_results_are_supplied(self, tmp_path):
        path = tmp_path / "utility_results.json"
        path.write_text(json.dumps({
            "experiments": {"synthetic -> real": {"map50": 0.3, "map50_95": 0.15}},
            "experiments_missing": [],
            "utility_retention_pct": 40.0,
            "fid": 88.1,
            "poisoning_verdict": "PASS",
        }))
        section = utility_section(path)
        assert section["status"] == "supplied"
        assert section["utility_retention_pct"] == 40.0
        assert section["experiments"]["synthetic -> real"]["map50_95"] == 0.15

    def test_missing_experiments_are_partial(self, tmp_path):
        path = tmp_path / "utility_results.json"
        path.write_text(json.dumps({"experiments": {}, "experiments_missing": ["real -> real"]}))
        assert utility_section(path)["status"] == "partially supplied"


class TestCalibrationAccounting:
    def test_log_written_before_the_correction_gives_the_true_total(self, tmp_path):
        from scripts.build_certificate import calibration_accounting

        (tmp_path / "calibration").mkdir()
        old_log = {"epsilon_calibration": 1.8, "queries": [{"query": f"q{i}"} for i in range(7)]}
        (tmp_path / "calibration" / "calibration_audit_log.json").write_text(json.dumps(old_log))
        accounting = calibration_accounting(tmp_path, 1.8)
        assert accounting["epsilon_per_query"] == 1.8
        assert accounting["epsilon_total"] == 12.6

    def test_without_a_log_the_configured_value_is_used(self, tmp_path):
        from scripts.build_certificate import calibration_accounting

        accounting = calibration_accounting(tmp_path, 1.8)
        assert accounting["epsilon_total"] == 12.6 and accounting["source"] == "configured default"
