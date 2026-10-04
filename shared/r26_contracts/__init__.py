"""Machine-checkable versions of the handoff contracts in INTERFACES.md."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator

__all__ = ["CONTRACTS", "ContractError", "schema", "validate", "validate_file"]


class ContractError(ValueError):
    """A handoff file does not match its contract."""

    def __init__(self, contract: str, source: str, problems: list[str]) -> None:
        self.contract, self.source, self.problems = contract, source, problems
        listed = "\n  ".join(problems)
        super().__init__(f"{source} does not match the '{contract}' contract:\n  {listed}")


def _schema_files() -> dict[str, str]:
    folder = resources.files(__package__) / "schemas"
    if not folder.is_dir():
        return {}
    return {
        entry.name.removesuffix(".schema.json"): entry.name
        for entry in folder.iterdir()
        if entry.name.endswith(".schema.json")
    }


CONTRACTS: tuple[str, ...] = tuple(sorted(_schema_files()))


@cache
def schema(contract: str) -> dict:
    files = _schema_files()
    if contract not in files:
        raise KeyError(f"unknown contract '{contract}'; known: {', '.join(CONTRACTS)}")
    text = (resources.files(__package__) / "schemas" / files[contract]).read_text(encoding="utf-8")
    return json.loads(text)


def _location(error) -> str:
    return "/".join(str(part) for part in error.absolute_path) or "(top level)"


def validate(document: object, contract: str, source: str = "document") -> None:
    """Raise ContractError listing every violation, or return quietly."""
    validator = Draft202012Validator(schema(contract))
    problems = [
        f"{_location(e)}: {e.message}"
        for e in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    ]
    if problems:
        raise ContractError(contract, source, problems)


def validate_file(path: Path, contract: str) -> dict:
    """Load a JSON handoff file, validate it, and return its contents."""
    if not path.is_file():
        raise ContractError(contract, str(path), ["file not found"])
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(contract, str(path), [f"not valid JSON: {exc}"]) from None
    validate(document, contract, path.name)
    return document
