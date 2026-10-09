"""Regulatory control mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControlMapping:
    provision: str
    requirement: str
    control: str
    owner: str
    evidence_check: str
    notes: str = ""

    def as_dict(self, verified: bool | None = None) -> dict:
        return {
            "provision": self.provision,
            "requirement": self.requirement,
            "control": self.control,
            "owner": self.owner,
            "evidence_check": self.evidence_check,
            "verified": verified,
            "notes": self.notes,
        }


MAPPINGS: tuple[ControlMapping, ...] = (
    ControlMapping(
        provision="GDPR Art. 5(1)(b)",
        requirement="Purpose limitation",
        control="Real statistics used only for simulator calibration, never reused for "
                "model training and never written to disk",
        owner="IT22309556",
        evidence_check="calibration_audit_log",
        notes="Audit log records seven aggregate queries and nothing else.",
    ),
    ControlMapping(
        provision="GDPR Art. 5(1)(c)",
        requirement="Data minimisation",
        control="Only seven aggregate fields extracted; per-town breakdowns and raw "
                "counts discarded before processing",
        owner="IT22309556",
        evidence_check="calibration_audit_log",
        notes="Compare the source file against calibration_stats.json.",
    ),
    ControlMapping(
        provision="GDPR Art. 5(2)",
        requirement="Accountability, the controller must be able to demonstrate compliance",
        control="HMAC-SHA256 signed audit reports, tamper-evident",
        owner="IT22309556",
        evidence_check="certificate_verification",
        notes="A report that can be silently edited demonstrates nothing.",
    ),
    ControlMapping(
        provision="GDPR Art. 17",
        requirement="Right to erasure",
        control="No individual record is stored anywhere in the pipeline",
        owner="All",
        evidence_check="k_anonymity",
        notes="The guarantee is structural rather than procedural.",
    ),
    ControlMapping(
        provision="GDPR Art. 25",
        requirement="Data protection by design and by default",
        control="Epsilon-DP enforced during calibration and during training, not applied "
                "to outputs afterwards",
        owner="IT22309556",
        evidence_check="budget_audit",
        notes="Per-step budget logs show epsilon accumulating during training.",
    ),
    ControlMapping(
        provision="GDPR Art. 25",
        requirement="Privacy budget accounting at calibration",
        control="Laplace mechanism on the seven calibration queries, with the total epsilon "
                "spent reported alongside the published statistics",
        owner="IT22309556",
        evidence_check="calibration_composition",
        notes="The queries' costs, recovered from the audit log, must compose to the "
              "reported total under sequential composition.",
    ),
    ControlMapping(
        provision="GDPR Art. 32",
        requirement="Security of processing",
        control="AES-256-CBC encryption, SHA-256 integrity hashing, role-based access control",
        owner="IT22541284",
        evidence_check="manifest_verification",
        notes="Digest verification across the delivered dataset.",
    ),
    ControlMapping(
        provision="GDPR Art. 32",
        requirement="Integrity of the data used in processing",
        control="Data poisoning detection before training: structural, per-batch "
                "statistical and cross-town checks, with rejected frames excluded",
        owner="IT22110220",
        evidence_check="poisoning_detection",
        notes="FILTERED counts as passing: rejected frames were removed before training.",
    ),
    ControlMapping(
        provision="PDPA s.5",
        requirement="Purpose limitation",
        control="As GDPR Art. 5(1)(b)",
        owner="IT22309556",
        evidence_check="calibration_audit_log",
    ),
    ControlMapping(
        provision="PDPA s.6",
        requirement="Data minimisation",
        control="As GDPR Art. 5(1)(c)",
        owner="IT22309556",
        evidence_check="calibration_audit_log",
    ),
)


DECLARED_WEAKNESSES: tuple[dict[str, str], ...] = (
    {
        "issue": "Opacus secure_mode disabled",
        "severity": "medium",
        "detail": "DP noise is drawn from a standard pseudo-random generator rather "
                  "than a cryptographically secure one.",
        "cause": "torchcsprng, which secure_mode needs, is unmaintained and only supports "
                 "PyTorch 1.8.1, so it cannot run with PyTorch 2.5 on any OS.",
        "remediation": "Use a secure noise source once one supports current PyTorch.",
        "source": "declared by IT22309556 in the privacy certificate",
    },
    {
        "issue": "Calibration total epsilon was understated, now corrected",
        "severity": "medium",
        "detail": "Seven calibration queries each ran at epsilon 1.8 and compose to 12.6 under "
                  "sequential composition; earlier outputs reported 1.8 as the total. The noise "
                  "was correct for the per-query value, so the published statistics stand.",
        "cause": "The total was set to the per-query value instead of the sum over queries.",
        "remediation": "Total now reported as 12.6 and checked by calibration_composition. "
                       "Running each query at 1.8/7 = 0.257 would restore a 1.8 total but needs "
                       "new statistics and a new capture.",
        "source": "found by the calibration_composition check; corrected by IT22309556",
    },
)
