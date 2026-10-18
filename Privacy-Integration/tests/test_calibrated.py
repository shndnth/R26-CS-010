"""The calibrated attack must find members that a plain loss threshold misses."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from privacy_integration.attack.calibrated import (
    calibrated_scores,
    evaluate_scores,
    reference_members,
    reference_pool,
    true_label_logits,
)
from privacy_integration.data.splits import stratified_split
from privacy_integration.settings import settings


class _Labelled(TensorDataset):
    @property
    def labels(self) -> list[int]:
        return [int(v) for v in self.tensors[1].view(-1)]


def _images(n: int, seed: int) -> _Labelled:
    g = torch.Generator().manual_seed(seed)
    return _Labelled(torch.randn(n, 3, 16, 16, generator=g), torch.randint(0, 2, (n, 1), generator=g).float())


def test_true_label_logit_flips_sign_for_negatives():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3, 1))
    with torch.no_grad():
        model[1].weight.fill_(0.0)
        model[1].bias.fill_(2.0)
    data = _Labelled(torch.zeros(2, 3), torch.tensor([[1.0], [0.0]]))
    assert np.allclose(true_label_logits(model, data, [0, 1], 2, "cpu"), [2.0, -2.0])


def test_calibration_removes_frame_difficulty():
    # Frames differ a lot in how easy they are. The target is only slightly better on its members,
    # so a raw threshold mostly sees difficulty, while calibration sees membership.
    rng = np.random.default_rng(0)
    n = 400
    difficulty = rng.normal(0, 3.0, 2 * n)
    refs = difficulty + rng.normal(0, 0.3, (4, 2 * n))
    target = difficulty + rng.normal(0, 0.3, 2 * n)
    target[:n] += 0.8
    raw = evaluate_scores(target[:n], target[n:]).auc
    cal = evaluate_scores(*np.split(calibrated_scores(target, refs), 2)).auc
    assert raw < 0.65 < 0.9 < cal


def test_reference_pool_never_contains_attacked_frames():
    pool = reference_pool([1, 2, 3, 4], [5, 6], exclude=[2, 5])
    assert pool == [1, 3, 4, 6]
    chosen = reference_members(list(range(100)), [i % 2 for i in range(100)], 20, 0)
    assert len(set(chosen)) <= 20 and set(chosen) <= set(range(100))


@pytest.mark.slow
def test_script_runs_resumes_and_matches_stress_targets(tmp_path, monkeypatch):
    from scripts import run_calibrated_stress, run_stress_test

    data = _images(240, seed=3)
    cfg = settings()

    def fake_load(args):
        return data, stratified_split(data.labels, 0.5, 0.2, 0), cfg, 0

    for module in (run_calibrated_stress, run_stress_test):
        monkeypatch.setattr(module, "load_dataset_and_split", fake_load)
        monkeypatch.setattr(module, "outputs_root", lambda: tmp_path)
    common = [
        "--data-dir",
        str(tmp_path),
        "--seeds",
        "7",
        "--n-members",
        "40",
        "--epochs",
        "15",
        "--epsilons",
        "1",
        "--bootstrap",
        "50",
        "--device",
        "cpu",
    ]

    monkeypatch.setattr(sys, "argv", ["pi-stress", *common])
    run_stress_test.main()
    monkeypatch.setattr(sys, "argv", ["pi-stress-calibrated", *common, "--n-ref", "2"])
    run_calibrated_stress.main()

    out = tmp_path / "stress_test" / "calibrated"
    report = json.loads((out / "calibrated_summary.json").read_text())
    assert {r["config"] for r in report["runs"]} == {"no_dp", "eps_1"}
    assert report["targets_match_stress_runs"] is True
    assert report["regime"]["n_reference_models"] == 2
    assert (out / "seed_7_refs.npz").is_file()
    no_dp = next(r for r in report["runs"] if r["config"] == "no_dp")
    assert no_dp["attack_auc"] > 0.6  # random labels: members are memorised, references cannot guess them

    stamp = (out / "seed_7_no_dp.json").stat().st_mtime
    run_calibrated_stress.main()  # rerun: everything is skipped
    assert (out / "seed_7_no_dp.json").stat().st_mtime == stamp
