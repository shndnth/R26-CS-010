"""Working folder layout and run lookup."""

from __future__ import annotations

import pytest

from utility_evaluation.config import Workspace


def test_weights_found_in_new_layout(tmp_path):
    path = tmp_path / "runs" / "kitti_baseline" / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    path.touch()
    assert Workspace(tmp_path).weights("kitti_baseline") == path


def test_weights_found_in_nested_legacy_layout(tmp_path):
    path = tmp_path / "runs" / "detect" / "runs" / "synthetic_baseline" / "weights" / "last.pt"
    path.parent.mkdir(parents=True)
    path.touch()
    assert Workspace(tmp_path).weights("synthetic_baseline", "last") == path


def test_missing_weights_lists_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="Searched"):
        Workspace(tmp_path).weights("nothing")


def test_folder_overrides(tmp_path):
    ws = Workspace(tmp_path, encrypted_dir=tmp_path / "enc", decrypted_dir=tmp_path / "shared",
                   kitti_dir=tmp_path / "k")
    assert ws.encrypted == tmp_path / "enc"
    assert ws.decrypted == tmp_path / "shared"
    assert ws.kitti_raw == tmp_path / "k"
    assert ws.yolo_dataset == tmp_path / "yolo_dataset"


def test_defaults_without_overrides(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.encrypted == tmp_path / "output"
    assert ws.decrypted == tmp_path / "decrypted_output"


def test_display_paths_are_relative_to_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.display(tmp_path / "runs" / "a" / "best.pt") == "runs/a/best.pt"
    assert ws.display(tmp_path.parent / "elsewhere" / "x.yaml") == "x.yaml"
