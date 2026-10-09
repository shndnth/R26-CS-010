"""Per-epsilon utility study: one detector per privacy configuration."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from utility_evaluation.config import CLASS_NAMES, SEED, SPLITS, TOWNS, Workspace
from utility_evaluation.conversion import build_yolo_dataset, collect_pairs
from utility_evaluation.training import TrainSettings

SYNTH_TO_SYNTH = "synthetic -> synthetic"
SYNTH_TO_REAL = "synthetic -> real"
REAL_TO_REAL = "real -> real"


class PlanError(ValueError):
    """The study plan is missing something or holds an invalid value."""


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    epsilon: float | None
    decrypted_dir: Path | None = None
    encrypted_dir: Path | None = None

    @property
    def label(self) -> str:
        return "no DP" if self.epsilon is None else f"epsilon {self.epsilon:g}"


@dataclass(frozen=True)
class StudyPlan:
    root: Path
    variants: tuple[Variant, ...]
    kitti_dataset: Path | None = None
    kitti_weights: Path | None = None
    training: TrainSettings = field(default_factory=TrainSettings)
    split_mode: str = "random"
    seed: int = SEED


def _path(value, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_plan(path: Path) -> StudyPlan:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent

    root = _path(raw.get("root"), base)
    if root is None:
        raise PlanError("root is required")

    variants = []
    for entry in raw.get("variants") or []:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise PlanError("every variant needs a name")
        decrypted = _path(entry.get("decrypted_dir"), base)
        encrypted = _path(entry.get("encrypted_dir"), base)
        if (decrypted is None) == (encrypted is None):
            raise PlanError(f"variant '{name}' needs exactly one of decrypted_dir or encrypted_dir")
        epsilon = entry.get("epsilon")
        variants.append(Variant(name, None if epsilon is None else float(epsilon), decrypted, encrypted))
    if not variants:
        raise PlanError("the plan lists no variants")

    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise PlanError("variant names must be unique")
    epsilons = [v.epsilon for v in variants]
    if len(set(epsilons)) != len(epsilons):
        raise PlanError("each epsilon may appear once")

    split_mode = raw.get("split_mode", "random")
    if split_mode not in ("random", "block"):
        raise PlanError("split_mode must be 'random' or 'block'")

    training = TrainSettings(**(raw.get("training") or {}))
    return StudyPlan(
        root=root,
        variants=tuple(sorted(variants, key=lambda v: (v.epsilon is None, v.epsilon or 0.0))),
        kitti_dataset=_path(raw.get("kitti_dataset"), base),
        kitti_weights=_path(raw.get("kitti_weights"), base),
        training=training,
        split_mode=split_mode,
        seed=int(raw.get("seed", SEED)),
    )


Trainer = Callable[[Path, Path, str, TrainSettings], Path]
Evaluator = Callable[..., dict]


def _default_trainer(data_yaml: Path, runs_dir: Path, name: str, settings: TrainSettings) -> Path:
    from utility_evaluation.training import train

    return train(data_yaml, runs_dir, name, settings)


def _default_evaluator(*args, **kwargs) -> dict:
    from utility_evaluation.detection import evaluate

    return evaluate(*args, **kwargs)


def _scores(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {"map50": record["map50"], "map50_95": record["map50_95"], "per_class": record["per_class"]}


class Study:
    def __init__(self, plan: StudyPlan, trainer: Trainer = _default_trainer,
                 evaluator: Evaluator = _default_evaluator, say: Callable[[str], None] = print) -> None:
        self.plan = plan
        self.trainer = trainer
        self.evaluator = evaluator
        self.say = say

    @property
    def output(self) -> Path:
        return self.plan.root / "epsilon_study.json"

    def workspace(self, variant: Variant) -> Workspace:
        root = self.plan.root / variant.name
        return Workspace(root=root, decrypted_dir=variant.decrypted_dir or root / "decrypted_output")

    def _kitti_yaml(self) -> Path | None:
        if self.plan.kitti_dataset is None:
            return None
        yaml_path = self.plan.kitti_dataset / "data.yaml"
        return yaml_path if yaml_path.is_file() else None

    def _decrypt(self, variant: Variant, ws: Workspace, force: bool) -> None:
        if variant.encrypted_dir is None:
            if not ws.decrypted.is_dir():
                raise FileNotFoundError(f"{ws.decrypted} not found for variant '{variant.name}'")
            return
        if ws.decrypted.is_dir() and not force:
            return
        from r26_common.env import dataset_aes_key

        from utility_evaluation.decryption import decrypt_town

        if ws.decrypted.exists():
            shutil.rmtree(ws.decrypted)
        key = dataset_aes_key()
        failed = 0
        for town in TOWNS:
            town_dir = variant.encrypted_dir / town
            if town_dir.is_dir():
                failed += decrypt_town(town_dir, ws.decrypted / town, key, self.say)["failed"]
        if failed:
            raise RuntimeError(f"{failed} file(s) in variant '{variant.name}' failed decryption or "
                               "manifest verification")

    def _gate(self, ws: Workspace) -> dict:
        from utility_evaluation.poisoning import audit, write_clean_labels

        result = audit(ws.decrypted, TOWNS)
        if not result.towns:
            raise FileNotFoundError(f"no towns with a labels/ folder under {ws.decrypted}")
        write_clean_labels(result, ws.decrypted, ws.clean_labels)
        ws.results.mkdir(parents=True, exist_ok=True)
        (ws.results / "poisoning_audit.json").write_text(json.dumps(result.as_dict(), indent=2),
                                                         encoding="utf-8")
        return result.as_dict()

    def run_variant(self, variant: Variant, force: bool = False) -> dict:
        ws = self.workspace(variant)
        self.say(f"\n=== {variant.name} ({variant.label}) ===")
        self._decrypt(variant, ws, force)

        audit = self._gate(ws)
        self.say(f"poisoning gate: {audit['verdict']}, {audit['frames_excluded']} frames excluded")

        weights = ws.runs / variant.name / "weights" / "best.pt"
        if force or not weights.is_file() or not (ws.yolo_dataset / "data.yaml").is_file():
            pairs = collect_pairs(ws.decrypted, TOWNS, ws.clean_labels)
            if not pairs:
                raise FileNotFoundError(f"no image and label pairs for variant '{variant.name}'")
            counts = build_yolo_dataset(pairs, ws.yolo_dataset, CLASS_NAMES, SPLITS["train"],
                                        SPLITS["val"], self.plan.seed, mode=self.plan.split_mode,
                                        overwrite=True)
            self.say(f"dataset: train={counts.train} val={counts.val} test={counts.test}")
            if (ws.runs / variant.name).exists():
                shutil.rmtree(ws.runs / variant.name)
            weights = self.trainer(ws.yolo_dataset / "data.yaml", ws.runs, variant.name, self.plan.training)
        else:
            self.say(f"reusing {ws.display(weights)}; pass --force to retrain")

        device = self.plan.training.device
        synthetic = self.evaluator(weights, ws.yolo_dataset / "data.yaml", SYNTH_TO_SYNTH, device,
                                   ws.runs / "val", f"{variant.name}_synthetic")
        kitti = self._kitti_yaml()
        real = (self.evaluator(weights, kitti, SYNTH_TO_REAL, device, ws.runs / "val",
                               f"{variant.name}_real") if kitti else None)

        return {
            "name": variant.name,
            "epsilon": variant.epsilon,
            "poisoning_verdict": audit["verdict"],
            "frames_scanned": audit["frames_scanned"],
            "frames_excluded": audit["frames_excluded"],
            SYNTH_TO_SYNTH: _scores(synthetic),
            SYNTH_TO_REAL: _scores(real),
        }

    def real_baseline(self) -> dict | None:
        kitti = self._kitti_yaml()
        weights = self.plan.kitti_weights
        if kitti is None or weights is None or not weights.is_file():
            return None
        self.say("\n=== KITTI baseline ===")
        return _scores(self.evaluator(weights, kitti, REAL_TO_REAL, self.plan.training.device,
                                      self.plan.root / "kitti_val", "kitti_baseline"))

    def run(self, only: list[str] | None = None, force: bool = False) -> dict:
        known = {v.name for v in self.plan.variants}
        unknown = set(only or []) - known
        if unknown:
            raise PlanError(f"unknown variants: {', '.join(sorted(unknown))}")

        previous = {}
        if self.output.is_file():
            previous = {r["name"]: r for r in json.loads(self.output.read_text(encoding="utf-8"))["variants"]}

        records = []
        for variant in self.plan.variants:
            if only and variant.name not in only:
                if variant.name in previous:
                    records.append(previous[variant.name])
                continue
            records.append(self.run_variant(variant, force))

        baseline = self.real_baseline()
        for record in records:
            record["utility_retention_pct"] = retention(record.get(SYNTH_TO_REAL), baseline)

        payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "training": {k: getattr(self.plan.training, k) for k in self.plan.training.__slots__},
            "split_mode": self.plan.split_mode,
            "seed": self.plan.seed,
            REAL_TO_REAL: baseline,
            "variants": sorted(records, key=lambda r: (r["epsilon"] is None, r["epsilon"] or 0.0)),
        }
        self.plan.root.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


def retention(synthetic_to_real: dict | None, real_to_real: dict | None) -> float | None:
    """Synthetic-to-real mAP50-95 as a percentage of the real-data baseline."""
    if not synthetic_to_real or not real_to_real:
        return None
    base = real_to_real["map50_95"]
    if not base or math.isclose(base, 0.0):
        return None
    return round(100.0 * synthetic_to_real["map50_95"] / base, 2)
