"""The stress test must detect leakage from an overfitted model and its removal by DP-SGD."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from privacy_integration.attack.stress import (
    NON_PRIVATE,
    StressRegime,
    StressRun,
    StressSummary,
    bootstrap_auc_ci,
    pick_members,
    run_one,
)
from privacy_integration.seeding import seed_everything
from privacy_integration.settings import settings


class _Labelled(TensorDataset):
    @property
    def labels(self) -> list[int]:
        return [int(v) for v in self.tensors[1].view(-1)]


def _random_images(n: int, seed: int) -> _Labelled:
    # Random images with random labels: nothing to generalise, so any attack signal is memorisation.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 16, 16, generator=g)
    y = torch.randint(0, 2, (n, 1), generator=g).float()
    return _Labelled(x, y)


def test_bootstrap_ci_brackets_the_auc():
    rng = np.random.default_rng(0)
    low, high = bootstrap_auc_ci(rng.normal(1, 1, 200), rng.normal(0, 1, 200), 200, 0)
    assert 0.5 < low < 0.76 < high < 0.9


def test_bootstrap_ci_covers_chance_for_identical_distributions():
    rng = np.random.default_rng(1)
    low, high = bootstrap_auc_ci(rng.normal(0, 1, 300), rng.normal(0, 1, 300), 300, 1)
    assert low < 0.5 < high


def test_pick_members_is_balanced_and_disjoint():
    members, nonmembers = pick_members(list(range(100)), list(range(100, 160)), 50, 3)
    assert len(members) == len(nonmembers) == 50
    assert not set(members) & set(nonmembers)


def test_pick_members_rejects_tiny_sets():
    with pytest.raises(ValueError):
        pick_members([1, 2, 3], [4, 5, 6], 50, 0)


def _run(config, seed, auc, ci):
    return StressRun(
        config, seed, None if config == NON_PRIVATE else 1.0, None, auc, ci, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5
    )


def test_verdict_requires_a_leaking_baseline():
    weak = StressSummary([_run(NON_PRIVATE, 1, 0.55, (0.52, 0.58)), _run("eps_1", 1, 0.50, (0.47, 0.53))])
    assert weak.verdict()["positive_control"] == "FAIL"
    assert weak.verdict()["dp_reduction"] == "NOT SHOWN"


def test_verdict_significant_when_cis_separate():
    runs = []
    for seed, (a, b) in zip((1, 2, 3), ((0.80, 0.51), (0.78, 0.50), (0.82, 0.52)), strict=True):
        runs += [
            _run(NON_PRIVATE, seed, a, (a - 0.04, a + 0.04)),
            _run("eps_1", seed, b, (b - 0.04, b + 0.04)),
        ]
    summary = StressSummary(runs)
    row = summary.compare()[0]
    assert row["ci95_disjoint_every_seed"] and row["p_value_one_sided"] < 0.01
    assert summary.verdict() == {
        **summary.verdict(),
        "positive_control": "PASS",
        "dp_reduction": "SIGNIFICANT",
    }


@pytest.mark.slow
def test_overfitted_baseline_leaks_and_dp_removes_it():
    data = _random_images(240, seed=0)
    members, nonmembers = list(range(0, 80)), list(range(80, 160))
    test_idx = list(range(160, 240))
    regime = StressRegime(n_members=80, epochs=30, batch_size=16, epsilons=(1.0,), bootstrap=200)
    cfg = settings()

    seed_everything(0)
    leak = run_one(data, data.labels, members, nonmembers, test_idx, None, 0, regime, "cpu", cfg)
    seed_everything(0)
    private = run_one(data, data.labels, members, nonmembers, test_idx, 1.0, 0, regime, "cpu", cfg)

    assert leak.attack_auc_ci[0] > 0.6, leak.as_dict()  # positive control: memorisation is visible
    assert private.attack_auc < leak.attack_auc - 0.15, private.as_dict()
    assert private.final_epsilon <= 1.0 + 1e-6


@pytest.mark.slow
def test_script_writes_summary(tmp_path, monkeypatch):
    import json
    import sys

    from privacy_integration.data.splits import stratified_split
    from scripts import run_stress_test

    data = _random_images(200, seed=1)
    cfg = settings()

    def fake_load(args):
        return data, stratified_split(data.labels, 0.5, 0.1, 0), cfg, 0

    monkeypatch.setattr(run_stress_test, "load_dataset_and_split", fake_load)
    monkeypatch.setattr(run_stress_test, "outputs_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-stress",
            "--data-dir",
            str(tmp_path),
            "--seeds",
            "5",
            "--n-members",
            "60",
            "--epochs",
            "5",
            "--epsilons",
            "3",
            "--bootstrap",
            "50",
            "--device",
            "cpu",
        ],
    )
    run_stress_test.main()
    report = json.loads((tmp_path / "stress_test" / "stress_test_summary.json").read_text())
    assert {r["config"] for r in report["runs"]} == {"no_dp", "eps_3"}
    assert report["comparisons"][0]["config"] == "eps_3"
    assert set(report["verdict"]) >= {"positive_control", "dp_reduction"}
