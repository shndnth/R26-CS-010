"""Stage definition, run workspace layout and the pipeline's own stages."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from r26_pipeline.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    root: Path

    @property
    def decrypted(self) -> Path:
        return self.root / "decrypted"

    @property
    def privacy(self) -> Path:
        return self.root / "privacy"

    @property
    def utility(self) -> Path:
        return self.root / "utility"

    @property
    def compliance(self) -> Path:
        return self.root / "compliance"

    @property
    def release(self) -> Path:
        return self.root / "release"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def state_file(self) -> Path:
        return self.root / "pipeline_state.json"

    @property
    def calibration(self) -> Path:
        return self.privacy / "calibration"

    @property
    def runs(self) -> Path:
        return self.privacy / "runs"

    @property
    def utility_results(self) -> Path:
        return self.utility / "outputs"

    @property
    def clean_labels(self) -> Path:
        return self.utility / "poisoning_detection_report" / "clean_labels"

    def utility_weights(self, run: str) -> Path:
        return self.utility / "runs" / run / "weights" / "best.pt"


# The calibration audit log is never released: published value minus logged noise is the real statistic.
RELEASE_FILES: tuple[tuple[str, str], ...] = (
    ("privacy/privacy_certificate.json", "privacy_certificate.json"),
    ("privacy/privacy_certificate_signature.json", "privacy_certificate_signature.json"),
    ("privacy/runs/summary.json", "privacy_summary.json"),
    ("privacy/calibration/calibration_stats.json", "calibration_stats.json"),
    ("utility/outputs/utility_results.json", "utility_results.json"),
    ("utility/outputs/poisoning_audit.json", "poisoning_audit.json"),
    ("utility/outputs/evaluations.json", "evaluations.json"),
    ("utility/outputs/fid.json", "fid.json"),
    ("compliance/compliance_report.json", "compliance_report.json"),
)
RELEASE_FIGURES = (("privacy/figures", "privacy"), ("utility/outputs/figures", "utility"))


Gate = Callable[[RunWorkspace], str | None]


@dataclass(frozen=True)
class Stage:
    name: str
    component: str
    summary: str
    module: str | None = None
    args: tuple[str, ...] = ()
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    contracts: tuple[tuple[Path, str], ...] = ()
    clean: tuple[Path, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    needs: tuple[str, ...] = ()
    skip_reason: str | None = None
    gate: Gate | None = None
    action: str | None = None
    extra: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def internal(self) -> bool:
        return self.component == "pipeline"

    def matches(self, selector: str) -> bool:
        return self.name == selector or self.name.startswith(selector + ".")


@dataclass(frozen=True)
class Context:
    cfg: PipelineConfig
    ws: RunWorkspace
    towns: list[str]
    encrypted: Path
    no_calibration: str | None
    no_kitti: str | None

    @classmethod
    def of(cls, cfg: PipelineConfig) -> Context:
        return cls(
            cfg=cfg,
            ws=RunWorkspace(cfg.workspace),
            towns=list(cfg.towns),
            encrypted=cfg.inputs.encrypted_dataset,
            no_calibration=None if cfg.inputs.calibration_source else "inputs.calibration_source is not set",
            no_kitti=None if cfg.inputs.kitti else "inputs.kitti is not set",
        )

    @property
    def every_run(self) -> list[Path]:
        p = self.cfg.privacy
        return [self.ws.runs / f"seed_{s}" / c for s in p.seeds for c in p.configs]

    @property
    def release_sources(self) -> tuple[Path, ...]:
        return (*(self.ws.root / source for source, _ in RELEASE_FILES),
                *(self.ws.root / folder for folder, _ in RELEASE_FIGURES))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def opt(flag: str, value) -> list[str]:
    return [] if value is None else [flag, str(value)]


def release(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "pipeline.release", "pipeline", "Collect the shareable deliverables into release/",
            inputs=ctx.release_sources,
            outputs=(ws.release / "privacy_certificate.json", ws.release / "utility_results.json",
                     ws.release / "compliance_report.json"),
            clean=(ws.release,),
            needs=("privacy.certificate", "compliance.report"),
            action="release",
        ),
    ]


def final(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "pipeline.final", "pipeline", "Integrated verdict and lineage",
            inputs=(ws.privacy / "privacy_certificate.json", ws.utility_results / "utility_results.json",
                    ws.compliance / "compliance_report.json"),
            outputs=(ws.release / "pipeline_report.json",),
            contracts=((ws.release / "pipeline_report.json", "pipeline_report"),),
            needs=("pipeline.release", "compliance.scan_leakage"),
            action="final",
        ),
    ]


def check_needs(stages: list[Stage]) -> None:
    names = {s.name for s in stages}
    for stage in stages:
        unknown = set(stage.needs) - names
        if unknown:
            raise RuntimeError(f"{stage.name} needs unknown stages: {unknown}")
