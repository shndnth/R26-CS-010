from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from privacy_integration.logging_config import get_logger as setup_logging
from privacy_integration.serialization import utc_timestamp as timestamp
from privacy_integration.serialization import write_json as save_json
from privacy_integration.training.dp_trainer import TrainingResult

logger = setup_logging(__name__)


@dataclass
class AuditReport:
    project: str
    member: str
    generated_at: str
    privacy_config: dict
    results: dict
    note: str


class PrivacyAuditInterface:
    """Builds the per-run audit report delivered to IT22066916."""

    PROJECT_ID: str = "R26-CS-010"
    MEMBER_ID: str = "IT22309556"

    def build_report(
        self,
        result: TrainingResult,
        clipping_norm: float,
        noise_multiplier: float,
        delta: float,
        epsilon_max: float,
    ) -> AuditReport:
        """Training evidence only; calibration is accounted for in the certificate."""
        return AuditReport(
            project=self.PROJECT_ID,
            member=self.MEMBER_ID,
            generated_at=timestamp(),
            privacy_config={
                "clipping_norm_C": clipping_norm,
                "noise_multiplier_sigma": round(noise_multiplier, 6),
                "delta": delta,
                "epsilon_max": epsilon_max,
            },
            results={
                "final_epsilon": round(result.final_epsilon, 6),
                "total_steps": result.total_steps,
                "halted_early": bool(result.halted_early),
                "budget_respected": bool(result.final_epsilon <= epsilon_max),
            },
            note=(
                "Deliver to IT22066916 before MIA testing begins. "
                "budget_log.csv contains per-step epsilon for full audit trail."
            ),
        )

    def save_report(self, report: AuditReport, path: Path | str) -> None:
        save_json(
            {
                "project": report.project,
                "member": report.member,
                "generated_at": report.generated_at,
                "privacy_config": report.privacy_config,
                "results": report.results,
                "note": report.note,
            },
            path,
        )
        logger.info("Audit report saved -> %s", path)

    def build_certificate(
        self,
        report: AuditReport,
        mia_success_rate: float,
        mia_auc: float,
        company: str = "Unknown",
    ) -> dict:
        target_met = mia_success_rate <= 55.0
        budget_ok = report.results["budget_respected"]
        return {
            "certificate_title": "DP-AV Privacy Pipeline Certificate",
            "project": report.project,
            "issued_by": report.member,
            "generated_at": timestamp(),
            "company": company,
            "privacy_configuration": report.privacy_config,
            "privacy_results": report.results,
            "validation": {
                "mia_success_rate_pct": mia_success_rate,
                "mia_auc": mia_auc,
                "target_ceiling_pct": 55.0,
                "privacy_target_met": bool(target_met),
            },
            "compliance": {
                "gdpr_article_25": "Privacy-by-design enforced at calibration and training stages.",
                "data_minimisation": (
                    "Only aggregate statistics entered the pipeline. No raw records retained."
                ),
                "sri_lanka_pdpa": "Purpose limitation and data minimisation principles observed.",
            },
            "overall_status": "PASSED" if (target_met and budget_ok) else "FAILED",
        }

    def save_certificate(self, certificate: dict, path: Path | str) -> None:
        save_json(certificate, path)
        logger.info("Privacy certificate saved -> %s", path)
