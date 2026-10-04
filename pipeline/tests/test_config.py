"""pipeline.yaml loading and validation."""

from __future__ import annotations

import pytest

from r26_pipeline.config import ConfigError, Environment, load_config


def write(tmp_path, body: str):
    path = tmp_path / "pipeline" / "pipeline.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


MINIMAL = """
workspace: ../../runs/main
repository: ..
inputs:
  encrypted_dataset: /data/output
"""


def test_minimal_config_resolves_paths(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.repository == tmp_path.resolve()
    assert cfg.workspace == (tmp_path.parent / "runs" / "main").resolve()
    assert cfg.inputs.calibration_source is None and cfg.inputs.kitti is None
    assert cfg.privacy.seeds == (42, 43, 44)
    assert cfg.utility.epochs == 50


def test_workspace_inside_repository_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="outside the repository"):
        load_config(write(tmp_path, MINIMAL.replace("../../runs/main", "../runs")))


def test_missing_dataset_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="encrypted_dataset"):
        load_config(write(tmp_path, "workspace: ../../w\ninputs: {}\n"))


def test_unknown_keys_are_refused(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(write(tmp_path, MINIMAL + "privacy:\n  seed: [1]\n"))


def test_invalid_split_mode(tmp_path):
    with pytest.raises(ConfigError, match="split_mode"):
        load_config(write(tmp_path, MINIMAL + "utility:\n  split_mode: town\n"))


def test_environment_forms(tmp_path, monkeypatch):
    cfg = load_config(write(tmp_path, MINIMAL + """
environments:
  privacy: research
  utility: {python: /envs/utility/bin/python}
  compliance: {}
"""))
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    assert cfg.environments["privacy"].command("-m", "x") == [
        "/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "research", "python", "-m", "x"]
    assert cfg.environments["utility"].command("-m", "x") == ["/envs/utility/bin/python", "-m", "x"]
    assert cfg.environments["compliance"].command("-m", "x")[1:] == ["-m", "x"]


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="pipeline.example.yaml"):
        load_config(tmp_path / "absent.yaml")


def test_environment_description():
    assert Environment("privacy", conda="research").describe() == "conda env 'research'"
