"""The contract machinery: schema validity, file loading, unknown names."""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from r26_contracts import CONTRACTS, ContractError, schema, validate_file


@pytest.mark.parametrize("name", CONTRACTS)
def test_every_schema_is_itself_valid(name):
    Draft202012Validator.check_schema(schema(name))


def test_validate_file_reports_missing_and_malformed(tmp_path):
    with pytest.raises(ContractError, match="not found"):
        validate_file(tmp_path / "absent.json", "utility_results")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ContractError, match="not valid JSON"):
        validate_file(bad, "utility_results")


def test_unknown_contract():
    with pytest.raises(KeyError):
        schema("nothing")
