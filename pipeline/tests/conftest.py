"""Shared fixtures: a stand-in repository whose components are tiny scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from r26_pipeline.config import COMPONENT_FOLDERS, Environment, Inputs, PipelineConfig

WRITE_JSON = '''
import json, sys
from pathlib import Path
target, payload = Path(sys.argv[1]), json.loads(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload))
print("wrote", target.name)
'''

FAIL = "import sys\nprint('broken on purpose')\nsys.exit(3)\n"


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for folder in COMPONENT_FOLDERS.values():
        scripts = repo / folder / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "__init__.py").write_text("")
        (scripts / "write_json.py").write_text(WRITE_JSON)
        (scripts / "fail.py").write_text(FAIL)
    return repo


@pytest.fixture
def config(tmp_path: Path, fake_repo: Path) -> PipelineConfig:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    return PipelineConfig(
        source=tmp_path / "pipeline.yaml",
        repository=fake_repo,
        workspace=tmp_path / "workspace",
        inputs=Inputs(dataset, None, None),
        environments={c: Environment(c) for c in ("privacy", "utility", "compliance")},
    )
