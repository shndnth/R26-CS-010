"""The pipeline as an ordered list of stages, one module per component."""

from __future__ import annotations

from r26_pipeline.config import PipelineConfig
from r26_pipeline.stages import privacy
from r26_pipeline.stages.base import (
    RELEASE_FIGURES,
    RELEASE_FILES,
    Context,
    Gate,
    RunWorkspace,
    Stage,
    check_needs,
)

__all__ = ["RELEASE_FIGURES", "RELEASE_FILES", "Gate", "RunWorkspace", "Stage", "build_stages"]


def build_stages(cfg: PipelineConfig) -> list[Stage]:
    ctx = Context.of(cfg)
    stages = [
        *privacy.calibration(ctx),
        *privacy.training(ctx),
        *privacy.certificate(ctx),
    ]
    check_needs(stages)
    return stages
