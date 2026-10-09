"""Risk and Compliance stages: manifest gate, independent checks, verdict, leakage scan."""

from __future__ import annotations

from r26_pipeline.stages.base import Context, RunWorkspace, Stage, read_json

CHECK_FILES = (
    "manifest_verification.json", "certificate_verification.json", "budget_audit.json",
    "reconstruction_attack.json", "calibration_composition.json", "anonymity.json",
)


def manifest_gate(ws: RunWorkspace) -> str | None:
    result = read_json(ws.compliance / "manifest_verification.json")
    if not result["all_valid"]:
        return ("the decrypted dataset does not match the generator's manifests; "
                "see manifest_verification.json. Nothing downstream may use it.")
    return None


def manifest(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "compliance.verify_manifest", "compliance",
            "Independent re-hash of the decrypted dataset against the manifests",
            module="scripts.verify_manifest",
            args=("--encrypted-dir", str(ctx.encrypted), "--decrypted-dir", str(ws.decrypted),
                  "--towns", *ctx.towns, "--output", str(ws.compliance / "manifest_verification.json")),
            inputs=(ws.decrypted,),
            outputs=(ws.compliance / "manifest_verification.json",),
            contracts=((ws.compliance / "manifest_verification.json", "manifest_verification"),),
            needs=("data.decrypt",),
            gate=manifest_gate,
        ),
    ]


def checks(ctx: Context) -> list[Stage]:
    ws, cfg = ctx.ws, ctx.cfg
    seeds = cfg.privacy.seeds
    calibration = (ws.calibration / "calibration_stats.json", ws.calibration / "calibration_audit_log.json")
    return [
        Stage(
            "compliance.verify_certs", "compliance",
            "Verify every signed audit report and run the tamper test",
            module="scripts.verify_certificates",
            args=("--runs-dir", str(ws.runs),
                  "--output", str(ws.compliance / "certificate_verification.json")),
            inputs=tuple(d / "audit_report.json" for d in ctx.every_run),
            outputs=(ws.compliance / "certificate_verification.json",),
            needs=tuple(f"privacy.train.seed{s}" for s in seeds),
        ),
        Stage(
            "compliance.audit_budgets", "compliance", "Reported against logged epsilon for every run",
            module="scripts.audit_budgets",
            args=("--runs-dir", str(ws.runs), "--output", str(ws.compliance / "budget_audit.json")),
            inputs=tuple(d / "audit_report.json" for d in ctx.every_run),
            outputs=(ws.compliance / "budget_audit.json",),
            needs=tuple(f"privacy.train.seed{s}" for s in seeds),
        ),
        Stage(
            "compliance.reconstruct", "compliance", "Reconstruction attack on the calibration statistics",
            module="scripts.reconstruction_attack",
            args=("--true-values", str(cfg.inputs.calibration_source),
                  "--published", str(calibration[0]),
                  "--audit-log", str(calibration[1]),
                  "--output", str(ws.compliance / "reconstruction_attack.json")),
            inputs=calibration,
            outputs=(ws.compliance / "reconstruction_attack.json",),
            needs=("privacy.calibrate",),
            skip_reason=ctx.no_calibration,
        ),
        Stage(
            "compliance.calibration_composition", "compliance",
            "Do the calibration queries compose to the claimed epsilon?",
            module="scripts.calibration_composition",
            args=("--audit-log", str(calibration[1]),
                  "--published", str(calibration[0]),
                  "--output", str(ws.compliance / "calibration_composition.json")),
            inputs=calibration,
            outputs=(ws.compliance / "calibration_composition.json",),
            contracts=((ws.compliance / "calibration_composition.json", "calibration_composition"),),
            needs=("privacy.calibrate",),
            skip_reason=ctx.no_calibration,
        ),
        Stage(
            "compliance.anonymity", "compliance", "k-anonymity of the released frame metadata",
            module="scripts.measure_anonymity",
            args=("--data-dir", str(ctx.encrypted), "--output", str(ws.compliance / "anonymity.json")),
            inputs=(ctx.encrypted,),
            outputs=(ws.compliance / "anonymity.json",),
        ),
        Stage(
            "compliance.report", "compliance", "Regulatory mapping and the compliance verdict",
            module="scripts.build_compliance_report",
            args=("--outputs-dir", str(ws.compliance), "--utility-dir", str(ws.utility_results),
                  "--output", str(ws.compliance / "compliance_report.json")),
            inputs=(*(ws.compliance / f for f in CHECK_FILES),
                    ws.utility_results / "poisoning_audit.json", ws.utility_results / "utility_results.json"),
            outputs=(ws.compliance / "compliance_report.json",),
            contracts=((ws.compliance / "compliance_report.json", "compliance_report"),),
            needs=("compliance.verify_manifest", "compliance.verify_certs", "compliance.audit_budgets",
                   "compliance.anonymity", "utility.report"),
        ),
    ]


def leakage(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "compliance.scan_leakage", "compliance",
            "Scan the release for paths, credentials and identifiers",
            module="scripts.scan_leakage",
            args=("--root", str(ws.release), "--output", str(ws.compliance / "leakage_scan.json")),
            inputs=ctx.release_sources,
            outputs=(ws.compliance / "leakage_scan.json",),
            needs=("pipeline.release",),
        ),
    ]
