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
    weak = StressSummary([_run(NON_PRIVATE, 1, 0.53, (0.48, 0.58)), _run("eps_1", 1, 0.50, (0.47, 0.53))])
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


def _leaky_but_below_old_cutoff() -> StressSummary:
    # Mirrors the first real run: every no_dp CI is above chance, one is below the old 0.60 cut-off,
    # and the eps_8 CIs overlap on some seeds while the paired drop is still significant.
    runs = []
    data = {1: (0.630, 0.484, 0.542), 2: (0.636, 0.495, 0.574), 3: (0.605, 0.499, 0.541)}
    for seed, (nd, e1, e8) in data.items():
        runs.append(_run(NON_PRIVATE, seed, nd, (nd - 0.036, nd + 0.035)))
        runs.append(_run("eps_1", seed, e1, (e1 - 0.036, e1 + 0.036)))
        runs.append(_run("eps_8", seed, e8, (e8 - 0.035, e8 + 0.035)))
    return StressSummary(runs)


def test_current_rules_use_standard_tests():
    verdict = _leaky_but_below_old_cutoff().verdict()
    assert verdict["positive_control"] == "PASS"
    assert verdict["dp_reduction"] == "SIGNIFICANT"
    assert verdict["first_version_rules"]["positive_control"] == "FAIL"


def test_reduction_not_shown_when_a_drop_is_not_significant():
    runs = [_run(NON_PRIVATE, s, 0.62, (0.58, 0.66)) for s in (1, 2, 3)]
    runs += [_run("eps_8", s, a, (a - 0.04, a + 0.04)) for s, a in ((1, 0.64), (2, 0.58), (3, 0.63))]
    assert StressSummary(runs).verdict()["dp_reduction"] == "NOT SHOWN"


def test_rescore_rebuilds_from_saved_report(tmp_path, monkeypatch):
    import json
    import sys

    from privacy_integration.attack.stress import from_report
    from scripts import rescore_stress_test

    summary = _leaky_but_below_old_cutoff()
    report = {
        "generated_at": "t0",
        "regime": {},
        "runs": [r.as_dict() for r in summary.runs],
        "verdict": {"positive_control": "FAIL", "dp_reduction": "NOT SHOWN"},
    }
    path = tmp_path / "stress_test_summary.json"
    path.write_text(json.dumps(report))

    assert from_report(report).verdict() == summary.verdict()
    monkeypatch.setattr(sys, "argv", ["pi-stress-rescore", "--summary", str(path)])
    rescore_stress_test.main()

    updated = json.loads(path.read_text())
    assert updated["verdict"]["dp_reduction"] == "SIGNIFICANT"
    assert updated["previous_verdicts"][0]["verdict"]["positive_control"] == "FAIL"
    assert all("significant" in c for c in updated["comparisons"])


def test_from_run_files_combines_seed_runs_from_different_days(tmp_path):
    import json

    from privacy_integration.attack.stress import from_run_files

    summary = _leaky_but_below_old_cutoff()
    for run in summary.runs:
        (tmp_path / f"seed_{run.seed}_{run.config}.json").write_text(json.dumps(run.as_dict()))
    (tmp_path / "stress_test_summary.json").write_text("{}")
    rebuilt = from_run_files(tmp_path)
    assert rebuilt.seeds() == [1, 2, 3]
    assert len(rebuilt.runs) == len(summary.runs)
    assert rebuilt.verdict() == summary.verdict()


def test_ordering_detects_stronger_privacy_weaker_attack():
    runs = []
    for seed, (e1, e8) in enumerate(((0.48, 0.54), (0.50, 0.57), (0.49, 0.55), (0.50, 0.55)), start=1):
        runs.append(_run(NON_PRIVATE, seed, 0.62, (0.58, 0.66)))
        r1 = _run("eps_1", seed, e1, (e1 - 0.03, e1 + 0.03))
        r8 = _run("eps_8", seed, e8, (e8 - 0.03, e8 + 0.03))
        r8.epsilon = 8.0
        runs += [r1, r8]
    row = StressSummary(runs).ordering()[0]
    assert (row["higher_epsilon"], row["lower_epsilon"]) == ("eps_8", "eps_1")
    assert row["significant"] and row["seeds"] == 4


def test_rescore_from_runs(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import rescore_stress_test

    summary = _leaky_but_below_old_cutoff()
    for run in summary.runs:
        (tmp_path / f"seed_{run.seed}_{run.config}.json").write_text(json.dumps(run.as_dict()))
    old = {
        "generated_at": "t0",
        "regime": {"epochs": 60},
        "runs": [summary.runs[0].as_dict()],
        "verdict": {"positive_control": "FAIL", "dp_reduction": "NOT SHOWN"},
    }
    path = tmp_path / "stress_test_summary.json"
    path.write_text(json.dumps(old))
    monkeypatch.setattr(sys, "argv", ["pi-stress-rescore", "--summary", str(path), "--from-runs"])
    rescore_stress_test.main()
    updated = json.loads(path.read_text())
    assert updated["regime"]["seeds"] == [1, 2, 3] and updated["regime"]["epochs"] == 60
    assert len(updated["runs"]) == 9 and updated["combined_from_run_files"]
    assert "epsilon_ordering" in updated
