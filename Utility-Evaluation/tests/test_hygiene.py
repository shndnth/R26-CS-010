"""Repository hygiene: no credentials or personal paths in source."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [p for d in ("utility_evaluation", "scripts") for p in (ROOT / d).rglob("*.py")]

RETIRED_KEY_DIGESTS = frozenset({
    "f12a12368f6e924e17fbc99aa778bb75173864e806fd4a71665ceedee59f43e5",
})
LITERAL = re.compile(r"""["']([^"'\n]{16,64})["']""")
ABSOLUTE_PATH = re.compile(r"""["'](?:[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/)""")


def test_no_retired_key_in_source():
    offenders = [
        p.name for p in SOURCES for m in LITERAL.finditer(p.read_text(encoding="utf-8"))
        if hashlib.sha256(m.group(1).encode()).hexdigest() in RETIRED_KEY_DIGESTS
    ]
    assert not offenders


def test_no_machine_specific_paths_in_source():
    offenders = [p.name for p in SOURCES if ABSOLUTE_PATH.search(p.read_text(encoding="utf-8"))]
    assert not offenders


def test_no_dotenv_file_in_component():
    assert not (ROOT / ".env").exists()
