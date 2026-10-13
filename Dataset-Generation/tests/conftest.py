"""Make the capture scripts importable without a CARLA installation."""

from __future__ import annotations

import sys
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

TEST_KEY = "0123456789abcdef0123456789abcdef"


def import_capture(monkeypatch, *argv):
    """Import capture_town with a stub carla module and the given command line."""
    monkeypatch.setitem(sys.modules, "carla", types.ModuleType("carla"))
    monkeypatch.setattr(sys, "argv", ["capture_town.py", *argv])
    monkeypatch.setenv("R26_DATASET_AES_KEY", TEST_KEY)
    sys.modules.pop("capture_town", None)
    import capture_town

    return capture_town
