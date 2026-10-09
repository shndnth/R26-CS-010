"""The real stage graph: ordering, optional inputs and skips."""

from __future__ import annotations

from dataclasses import replace

from r26_pipeline.config import Inputs, PrivacySettings
from r26_pipeline.runner import Pipeline
from r26_pipeline.stages import build_stages


def test_stage_names_are_unique_and_ordered_after_their_needs(config):
    stages = build_stages(config)
    names = [s.name for s in stages]
    assert len(names) == len(set(names))
    for position, stage in enumerate(stages):
        for need in stage.needs:
            assert names.index(need) < position, f"{stage.name} runs before {need}"


def test_one_training_and_attack_stage_per_seed(config):
    cfg = replace(config, privacy=PrivacySettings(seeds=(1, 2)))
    names = [s.name for s in build_stages(cfg)]
    expected = {"privacy.train.seed1", "privacy.train.seed2", "privacy.attack.seed1", "privacy.attack.seed2"}
    assert expected <= set(names)


def test_missing_optional_inputs_skip_their_stages_and_dependents(config):
    skips = Pipeline(config, echo=False).skips
    assert {"privacy.calibrate", "compliance.reconstruct", "compliance.calibration_composition"} <= set(skips)
    assert {"utility.build_kitti", "utility.train_kitti", "utility.fid"} <= set(skips)
    assert "utility.cross_eval" not in skips and "compliance.report" not in skips


def test_optional_inputs_enable_every_stage(config, tmp_path):
    inputs = Inputs(config.inputs.encrypted_dataset, tmp_path / "true.json", tmp_path / "kitti")
    cfg = replace(config, inputs=inputs)
    assert Pipeline(cfg, echo=False).skips == {}


def test_privacy_training_reads_the_poisoning_gate_output(config):
    train = next(s for s in build_stages(config) if s.name.startswith("privacy.train"))
    assert "--labels-dir" in train.args
    assert "utility.detect_poisoning" in train.needs


def test_privacy_stages_write_into_the_workspace(config):
    for stage in build_stages(config):
        if stage.component == "privacy":
            assert dict(stage.env)["R26_OUTPUTS_DIR"].startswith(str(config.workspace))


def test_release_never_contains_the_calibration_audit_log():
    from r26_pipeline.stages import RELEASE_FILES

    assert not any("audit_log" in source for source, _ in RELEASE_FILES)


def test_selection_by_prefix(config):
    pipeline = Pipeline(config, echo=False)
    names = [s.name for s in pipeline.select(only=["privacy.train"])]
    assert names == ["privacy.train.seed42", "privacy.train.seed43", "privacy.train.seed44"]
    window = [s.name for s in pipeline.select(start="utility.report", until="pipeline.release")]
    assert window[0] == "utility.report" and window[-1] == "pipeline.release"
