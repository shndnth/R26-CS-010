"""Shared fixtures. Builds artefacts matching the Privacy Integration formats."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("R26_AUDIT_HMAC_KEY", "test-hmac-key-not-for-production")

TEST_KEY = b"test-hmac-key-not-for-production"


def sign(payload: str, key: bytes = TEST_KEY) -> dict:
    return {
        "project": "R26-CS-010",
        "member": "IT22309556",
        "report_file": "audit_report.json",
        "sha256_content": hashlib.sha256(payload.encode()).hexdigest(),
        "hmac_signature": hmac.new(key, payload.encode(), hashlib.sha256).hexdigest(),
        "algorithm": "HMAC-SHA256",
    }


def make_run(
    root: Path,
    config: str,
    final_epsilon: float | None = 2.9988,
    steps: int = 100,
    private: bool = True,
    ceiling_breach: bool = False,
) -> Path:
    """Create one run directory in the Privacy Integration layout."""
    run_dir = root / config
    run_dir.mkdir(parents=True, exist_ok=True)

    if private:
        report = {
            "project": "R26-CS-010",
            "target_epsilon": 3.0,
            "results": {
                "final_epsilon": final_epsilon,
                "halted_early": False,
                "budget_respected": True,
            },
        }
        peak = 12.0 if ceiling_breach else final_epsilon
        rows = ["step,epoch,cumulative_eps,loss"]
        for i in range(steps):
            eps = peak * (i + 1) / steps
            rows.append(f"{i},0,{eps:.6f},0.5")
        (run_dir / "budget_log.csv").write_text("\n".join(rows), encoding="utf-8")
    else:
        report = {
            "differential_privacy_applied": False,
            "config": config,
            "privacy_config": {"epsilon": None},
            "results": {"halted_early": False},
            "warning": "absence of epsilon is intentional and MUST NOT be read as zero",
        }

    payload = json.dumps(report, indent=2)
    (run_dir / "audit_report.json").write_text(payload, encoding="utf-8")
    (run_dir / "audit_certificate.json").write_text(
        json.dumps(sign(payload), indent=2), encoding="utf-8"
    )
    return run_dir


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "runs" / "seed_42"
    make_run(root, "config_b", final_epsilon=2.9988, steps=100)
    make_run(root, "no_dp", private=False)
    return tmp_path / "runs"
