"""Regulatory mapping structure."""

from __future__ import annotations

from risk_compliance.mapping import DECLARED_WEAKNESSES, MAPPINGS


class TestMappings:
    def test_covers_the_expected_provisions(self):
        provisions = {m.provision for m in MAPPINGS}
        for expected in (
            "GDPR Art. 5(1)(b)", "GDPR Art. 5(1)(c)", "GDPR Art. 5(2)",
            "GDPR Art. 17", "GDPR Art. 25", "GDPR Art. 32",
        ):
            assert expected in provisions

    def test_covers_sri_lanka_pdpa(self):
        provisions = {m.provision for m in MAPPINGS}
        assert any(p.startswith("PDPA") for p in provisions)

    def test_every_mapping_names_an_owner(self):
        assert all(m.owner for m in MAPPINGS)

    def test_every_mapping_links_to_an_evidence_check(self):
        """A provision without evidence is an assertion, not a verification."""
        assert all(m.evidence_check for m in MAPPINGS)

    def test_as_dict_records_verification_state(self):
        entry = MAPPINGS[0].as_dict(verified=True)
        assert entry["verified"] is True
        assert entry["provision"] == MAPPINGS[0].provision

    def test_unverified_state_is_distinct_from_failed(self):
        assert MAPPINGS[0].as_dict(verified=None)["verified"] is None
        assert MAPPINGS[0].as_dict(verified=False)["verified"] is False


class TestDeclaredWeaknesses:
    def test_weaknesses_are_declared(self):
        assert len(DECLARED_WEAKNESSES) >= 2

    def test_each_has_severity_and_remediation(self):
        for weakness in DECLARED_WEAKNESSES:
            assert weakness["severity"] in {"low", "medium", "high"}
            assert weakness["remediation"]

    def test_calibration_total_correction_is_recorded(self):
        [weakness] = [w for w in DECLARED_WEAKNESSES if "calibration total" in w["issue"].lower()]
        assert "12.6" in weakness["detail"]
        assert "0.257" in weakness["remediation"]
