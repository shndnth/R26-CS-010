"""Repository-wide rules: no secrets, no docs folder, no em or en dashes."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg", ".ini", ".example", ""}
DASHES = {chr(0x2014): "em dash", chr(0x2013): "en dash"}


def tracked_files():
    for path in REPOSITORY.rglob("*"):
        if path.is_file() and not SKIP_PARTS & set(path.relative_to(REPOSITORY).parts) \
                and not any(p.endswith(".egg-info") for p in path.parts):
            yield path


def test_no_em_or_en_dashes_anywhere():
    offenders = []
    for path in tracked_files():
        if path.suffix not in TEXT_SUFFIXES and not path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            for char, name in DASHES.items():
                if char in line:
                    offenders.append(f"{path.relative_to(REPOSITORY)}:{number} {name}")
    assert not offenders, "\n".join(offenders[:50])


def test_no_docs_folder():
    assert not (REPOSITORY / "docs").exists()


def test_no_environment_files_with_values():
    for path in tracked_files():
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            raise AssertionError(f"{path} must not be in the repository")


def test_env_examples_hold_no_values():
    for path in REPOSITORY.rglob(".env.example"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                assert line.rstrip().endswith("="), f"{path}: {line}"
