"""Steps that run inside the pipeline: assembling the release and the integrated verdict."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from r26_common.hashing import sha256_file

from r26_pipeline.stages import RELEASE_FIGURES, RELEASE_FILES, RunWorkspace, Stage
from r26_pipeline.state import RunState, now


def _relative(ws: RunWorkspace, path: str) -> str:
    try:
        return Path(path).relative_to(ws.root).as_posix()
    except ValueError:
        return Path(path).name


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_manifest(release: Path) -> None:
    files = sorted(p for p in release.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = {p.relative_to(release).as_posix(): sha256_file(p) for p in files}
    (release / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def assemble_release(ws: RunWorkspace) -> list[str]:
    ws.release.mkdir(parents=True, exist_ok=True)
    copied = []
    for source, target in RELEASE_FILES:
        path = ws.root / source
        if path.is_file():
            shutil.copy2(path, ws.release / target)
            copied.append(target)
    for folder, prefix in RELEASE_FIGURES:
        for figure in sorted((ws.root / folder).glob("*.png")):
            destination = ws.release / "figures" / f"{prefix}_{figure.name}"
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(figure, destination)
            copied.append(destination.relative_to(ws.release).as_posix())
    write_manifest(ws.release)
    return copied


def _privacy_section(ws: RunWorkspace) -> dict:
    certificate = _read(ws.privacy / "privacy_certificate.json") or {}
    summary = _read(ws.runs / "summary.json") or {}
    verdicts = certificate.get("certification", {})
    return {
        "verdict": verdicts.get("overall_privacy_verdict", "MISSING"),
        "privacy_enforcement": verdicts.get("privacy_enforcement"),
        "empirical_validation": verdicts.get("empirical_validation"),
        "configurations": {
            tag: {
                "target_epsilon": c.get("target_epsilon"),
                "seeds": c.get("n_seeds"),
                "utility_auc_mean": c.get("utility_auc_mean"),
                "attack_auc_mean": c.get("attack_auc_mean"),
            }
            for tag, c in summary.get("configs", {}).items()
        },
    }


def _utility_section(ws: RunWorkspace) -> dict:
    results = _read(ws.utility_results / "utility_results.json")
    if results is None:
        return {"status": "MISSING"}
    missing = results.get("experiments_missing") or []
    return {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "utility_retention_pct": results.get("utility_retention_pct"),
        "sim_to_real_gap_map50_95": results.get("sim_to_real_gap_map50_95"),
        "fid": results.get("fid"),
        "poisoning_verdict": results.get("poisoning_verdict"),
        "experiments": {k: v.get("map50_95") for k, v in results.get("experiments", {}).items()},
        "experiments_missing": missing,
    }


def _compliance_section(ws: RunWorkspace) -> dict:
    report = _read(ws.compliance / "compliance_report.json") or {}
    leakage = _read(ws.compliance / "leakage_scan.json") or {}
    return {
        "release_leakage_scan": leakage.get("verdict", "not run"),
        "release_leakage_findings": leakage.get("by_issue", {}),
        "verdict": report.get("overall_verdict", "MISSING"),
        "checks_not_run": report.get("checks_not_run", []),
        "checks_failed": sorted({
            m["evidence_check"] for m in report.get("regulatory_mapping", []) if m.get("verified") is False
        }),
        "provisions_verified": sum(1 for m in report.get("regulatory_mapping", []) if m.get("verified")),
        "provisions_total": len(report.get("regulatory_mapping", [])),
        "declared_weaknesses": len(report.get("declared_weaknesses", [])),
    }


def overall_verdict(privacy: str, utility: str, compliance: str, skipped: list[str]) -> str:
    if "FAIL" in (privacy, compliance):
        return "FAIL"
    if privacy == "PASS" and compliance == "PASS" and utility == "COMPLETE" and not skipped:
        return "PASS"
    return "INCOMPLETE"


def final_report(ws: RunWorkspace, stages: list[Stage], state: RunState, project: str = "R26-CS-010") -> dict:
    privacy, utility, compliance = _privacy_section(ws), _utility_section(ws), _compliance_section(ws)
    lineage = []
    for stage in stages:
        record = state.get(stage.name)
        if stage.name == "pipeline.final":
            continue
        lineage.append({
            "stage": stage.name,
            "component": stage.component,
            "status": record.status if record else "not run",
            "finished": record.finished if record else None,
            "seconds": record.seconds if record else None,
            "detail": record.detail if record else stage.skip_reason,
            "code": record.code if record else None,
            "outputs": {_relative(ws, p): d for p, d in (record.outputs if record else {}).items()},
        })
    skipped = [entry["stage"] for entry in lineage if entry["status"] != "done"]

    report = {
        "project": project,
        "generated_at": now(),
        "overall_verdict": overall_verdict(
            privacy["verdict"], utility["status"], compliance["verdict"], skipped
        ),
        "verdict_rule": (
            "PASS needs a PASS privacy certificate, a PASS compliance report, all four utility "
            "experiments, and every stage run. FAIL if privacy or compliance failed. Anything "
            "else is INCOMPLETE, which is not a failure."
        ),
        "privacy": privacy,
        "utility": utility,
        "compliance": compliance,
        "stages_not_completed": skipped,
        "lineage": lineage,
    }

    ws.release.mkdir(parents=True, exist_ok=True)
    (ws.release / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_manifest(ws.release)
    return report


def print_summary(report: dict) -> None:
    privacy, utility, compliance = report["privacy"], report["utility"], report["compliance"]
    print("\n" + "=" * 64)
    print(f"R26-CS-010 pipeline   overall: {report['overall_verdict']}")
    print("=" * 64)
    print(f"privacy      {privacy['verdict']:<11} enforcement {privacy['privacy_enforcement']}, "
          f"validation {privacy['empirical_validation']}")
    for tag, c in privacy["configurations"].items():
        eps = "none" if c["target_epsilon"] is None else f"{c['target_epsilon']:g}"
        attack = "n/a" if c["attack_auc_mean"] is None else f"{c['attack_auc_mean']:.4f}"
        print(f"   {tag:<10} epsilon {eps:<5} utility AUC {c['utility_auc_mean']:.4f}  attack AUC {attack}")
    retention = utility.get("utility_retention_pct")
    print(f"utility      {utility['status']:<11} retention "
          f"{'n/a' if retention is None else f'{retention:.1f}%'}, FID {utility.get('fid')}, "
          f"poisoning {utility.get('poisoning_verdict')}")
    if utility.get("experiments_missing"):
        print(f"   not run: {', '.join(utility['experiments_missing'])}")
    print(f"compliance   {compliance['verdict']:<11} {compliance['provisions_verified']}/"
          f"{compliance['provisions_total']} provisions verified")
    if compliance["checks_failed"]:
        print(f"   checks failed: {', '.join(compliance['checks_failed'])}")
    if compliance["checks_not_run"]:
        print(f"   checks not run: {', '.join(compliance['checks_not_run'])}")
    if report["stages_not_completed"]:
        print(f"stages not completed: {', '.join(report['stages_not_completed'])}")
