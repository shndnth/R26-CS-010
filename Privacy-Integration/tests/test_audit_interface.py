import unittest

from privacy_integration.audit.report import AuditReport, PrivacyAuditInterface
from privacy_integration.training.dp_trainer import StepRecord, TrainingResult


def make_training_result(
    final_epsilon: float = 2.9,
    total_steps: int = 100,
    halted_early: bool = False,
) -> TrainingResult:
    result = TrainingResult()
    result.final_epsilon = final_epsilon
    result.noise_multiplier = 1.01
    result.total_steps = total_steps
    result.halted_early = halted_early
    result.budget_log = [
        StepRecord(step=i, epoch=0, cumulative_eps=round(i * 0.03, 4), loss=0.7)
        for i in range(total_steps)
    ]
    return result


class TestPrivacyAuditInterface(unittest.TestCase):

    def setUp(self):
        self.interface = PrivacyAuditInterface()
        self.training_result = make_training_result()
        self.report = self.interface.build_report(
            result=self.training_result,
            clipping_norm=1.0,
            noise_multiplier=1.01,
            delta=1e-5,
            epsilon_max=10.0,
        )

    def test_report_is_audit_report_instance(self):
        self.assertIsInstance(self.report, AuditReport)

    def test_project_id_correct(self):
        self.assertEqual(self.report.project, "R26-CS-010")

    def test_member_id_correct(self):
        self.assertEqual(self.report.member, "IT22309556")

    def test_reports_training_epsilon_only(self):
        self.assertAlmostEqual(self.report.results["final_epsilon"], 2.9, places=5)
        self.assertNotIn("total_epsilon", self.report.results)
        self.assertNotIn("calibration_epsilon", self.report.results)

    def test_budget_respected_true_when_within_ceiling(self):
        self.assertTrue(self.report.results["budget_respected"])

    def test_budget_respected_false_when_exceeded(self):
        result = make_training_result(final_epsilon=11.0)
        report = self.interface.build_report(
            result=result,
            clipping_norm=1.0,
            noise_multiplier=0.67,
            delta=1e-5,
            epsilon_max=10.0,
        )
        self.assertFalse(report.results["budget_respected"])

    def test_halted_early_serialises_as_bool(self):
        self.assertIsInstance(self.report.results["halted_early"], bool)

    def test_certificate_passes_when_mia_below_ceiling(self):
        cert = self.interface.build_certificate(
            report=self.report,
            mia_success_rate=52.0,
            mia_auc=0.51,
            company="Test Corp",
        )
        self.assertEqual(cert["overall_status"], "PASSED")

    def test_certificate_fails_when_mia_above_ceiling(self):
        cert = self.interface.build_certificate(
            report=self.report,
            mia_success_rate=68.0,
            mia_auc=0.71,
        )
        self.assertEqual(cert["overall_status"], "FAILED")

    def test_certificate_contains_gdpr_compliance_section(self):
        cert = self.interface.build_certificate(
            report=self.report,
            mia_success_rate=51.0,
            mia_auc=0.50,
        )
        self.assertIn("compliance", cert)
        self.assertIn("gdpr_article_25", cert["compliance"])

    def test_generated_at_present_in_report(self):
        self.assertIsNotNone(self.report.generated_at)


if __name__ == "__main__":
    unittest.main()
