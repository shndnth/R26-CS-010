"""Per-epsilon study orchestration, with training and evaluation stood in."""

from __future__ import annotations

import shutil

import pytest

from r26_contracts import validate_file
from tests.test_pipeline import KEY, build_encrypted_dataset
from utility_evaluation.study import (
    REAL_TO_REAL,
    SYNTH_TO_REAL,
    SYNTH_TO_SYNTH,
    PlanError,
    Study,
    load_plan,
    retention,
)

SCORES = {SYNTH_TO_SYNTH: 0.60, SYNTH_TO_REAL: 0.20, REAL_TO_REAL: 0.40}


class Recorder:
    """Stands in for YOLO: records calls and returns fixed scores."""

    def __init__(self):
        self.trained, self.evaluated = [], []

    def trainer(self, data_yaml, runs_dir, name, settings):
        weights = runs_dir / name / "weights" / "best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"weights")
        self.trained.append((name, data_yaml))
        return weights

    def evaluator(self, weights, data_yaml, label, device, project, name, **extra):
        self.evaluated.append((label, name))
        return {"label": label, "map50": SCORES[label] + 0.2, "map50_95": SCORES[label],
                "per_class": {"car": None}}


def write_plan(tmp_path, variants, extra=""):
    lines = "\n".join(f"  - {v}" for v in variants)
    plan = tmp_path / "plan.yaml"
    plan.write_text(f"root: ./study\ntraining: {{epochs: 1}}\n{extra}variants:\n{lines}\n")
    return plan


@pytest.fixture
def encrypted(tmp_path, monkeypatch):
    monkeypatch.setenv("R26_DATASET_AES_KEY", KEY.decode())
    build_encrypted_dataset(tmp_path / "eps1")
    shutil.copytree(tmp_path / "eps1", tmp_path / "eps8")
    return tmp_path


def test_plan_resolves_paths_and_orders_variants(tmp_path):
    plan = load_plan(write_plan(tmp_path, [
        "{name: no_dp, epsilon:, decrypted_dir: ./d}",
        "{name: eps_8, epsilon: 8, encrypted_dir: ./e8}",
        "{name: eps_1, epsilon: 1, encrypted_dir: ./e1}",
    ]))
    assert [v.name for v in plan.variants] == ["eps_1", "eps_8", "no_dp"]
    assert plan.root == (tmp_path / "study").resolve()
    assert plan.training.epochs == 1


@pytest.mark.parametrize(("variants", "message"), [
    (["{name: a, epsilon: 1}"], "exactly one"),
    (["{name: a, epsilon: 1, decrypted_dir: x}", "{name: a, epsilon: 3, decrypted_dir: y}"], "unique"),
    (["{name: a, epsilon: 1, decrypted_dir: x}", "{name: b, epsilon: 1, decrypted_dir: y}"], "once"),
])
def test_invalid_plans(tmp_path, variants, message):
    with pytest.raises(PlanError, match=message):
        load_plan(write_plan(tmp_path, variants))


def test_study_runs_every_variant_and_computes_retention(encrypted):
    kitti = encrypted / "kitti_dataset"
    kitti.mkdir()
    (kitti / "data.yaml").write_text("names: {}\n")
    (encrypted / "kitti.pt").write_bytes(b"w")
    plan = load_plan(write_plan(encrypted, [
        "{name: eps_1, epsilon: 1, encrypted_dir: ./eps1/output}",
        "{name: eps_8, epsilon: 8, encrypted_dir: ./eps8/output}",
    ], extra="kitti_dataset: ./kitti_dataset\nkitti_weights: ./kitti.pt\n"))
    recorder = Recorder()
    payload = Study(plan, recorder.trainer, recorder.evaluator, say=lambda _m: None).run()

    assert [r["name"] for r in payload["variants"]] == ["eps_1", "eps_8"]
    assert {name for name, _ in recorder.trained} == {"eps_1", "eps_8"}
    record = payload["variants"][0]
    assert record["poisoning_verdict"] == "PASS"
    assert record[SYNTH_TO_REAL]["map50_95"] == 0.20
    assert record["utility_retention_pct"] == 50.0
    assert payload[REAL_TO_REAL]["map50_95"] == 0.40
    saved = validate_file(plan.root / "epsilon_study.json", "epsilon_study")
    assert saved["variants"][1]["epsilon"] == 8.0
    assert (plan.root / "eps_1" / "decrypted_output" / "town01" / "rgb_f000000.png").is_file()


def test_trained_variants_are_reused_and_only_reruns_one(encrypted):
    plan = load_plan(write_plan(encrypted, [
        "{name: eps_1, epsilon: 1, encrypted_dir: ./eps1/output}",
        "{name: eps_8, epsilon: 8, encrypted_dir: ./eps8/output}",
    ]))
    first = Recorder()
    Study(plan, first.trainer, first.evaluator, say=lambda _m: None).run()

    second = Recorder()
    payload = Study(plan, second.trainer, second.evaluator, say=lambda _m: None).run(only=["eps_8"])
    assert second.trained == []
    assert [r["name"] for r in payload["variants"]] == ["eps_1", "eps_8"]
    assert payload[REAL_TO_REAL] is None
    assert payload["variants"][0]["utility_retention_pct"] is None

    third = Recorder()
    Study(plan, third.trainer, third.evaluator, say=lambda _m: None).run(only=["eps_8"], force=True)
    assert [name for name, _ in third.trained] == ["eps_8"]


def test_unknown_variant_is_rejected(encrypted):
    plan = load_plan(write_plan(encrypted, ["{name: eps_1, epsilon: 1, encrypted_dir: ./eps1/output}"]))
    with pytest.raises(PlanError, match="unknown"):
        Study(plan, say=lambda _m: None).run(only=["eps_3"])


def test_retention():
    assert retention({"map50_95": 0.2}, {"map50_95": 0.4}) == 50.0
    assert retention(None, {"map50_95": 0.4}) is None
    assert retention({"map50_95": 0.2}, {"map50_95": 0.0}) is None
