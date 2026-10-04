"""Privacy Integration stages: calibration, DP-SGD training, attacks, certificate."""

from __future__ import annotations

from r26_pipeline.stages.base import Context, Gate, RunWorkspace, Stage, opt, read_json


def seeds_gate(expected: tuple[int, ...]) -> Gate:
    def gate(ws: RunWorkspace) -> str | None:
        summary = read_json(ws.runs / "summary.json")
        found = {s for cfg in summary["configs"].values() for s in cfg.get("seeds", [])}
        extra = found - set(expected)
        if extra:
            return (f"runs/ holds seeds {sorted(extra)} that are not in pipeline.yaml, and they were "
                    f"aggregated into summary.json. Delete those seed folders or add them to privacy.seeds.")
        return None
    return gate


def _env(ctx: Context) -> tuple[tuple[str, str], ...]:
    return (("R26_OUTPUTS_DIR", str(ctx.ws.privacy)),)


def calibration(ctx: Context) -> list[Stage]:
    cfg, ws = ctx.cfg, ctx.ws
    return [
        Stage(
            "privacy.calibrate", "privacy",
            "Laplace mechanism on the real calibration statistics",
            module="scripts.run_calibration",
            args=(
                "--input", str(cfg.inputs.calibration_source),
                "--output-dir", str(ws.calibration),
                *opt("--epsilon", cfg.privacy.calibration_epsilon),
            ),
            inputs=tuple(x for x in (cfg.inputs.calibration_source,) if x),
            outputs=(
                ws.calibration / "calibration_stats.json",
                ws.calibration / "calibration_audit_log.json",
            ),
            contracts=(
                (ws.calibration / "calibration_stats.json", "calibration_stats"),
                (ws.calibration / "calibration_audit_log.json", "calibration_audit_log"),
            ),
            env=_env(ctx),
            skip_reason=ctx.no_calibration,
        ),
    ]


def training(ctx: Context) -> list[Stage]:
    ws, p = ctx.ws, ctx.cfg.privacy
    shared = ["--data-dir", str(ws.decrypted), "--labels-dir", str(ws.clean_labels),
              "--towns", *ctx.towns, *opt("--device", p.device)]
    stages: list[Stage] = []
    for seed in p.seeds:
        run_dirs = [ws.runs / f"seed_{seed}" / c for c in p.configs]
        stages.append(Stage(
            f"privacy.train.seed{seed}", "privacy",
            f"DP-SGD training, seed {seed}: {', '.join(p.configs)}",
            module="scripts.run_training",
            args=(*shared, "--seed", str(seed), "--config", *p.configs,
                  *opt("--epochs", p.epochs)),
            inputs=(ws.decrypted, ws.clean_labels),
            outputs=tuple(d / f for d in run_dirs for f in ("model.pt", "metrics.json", "audit_report.json")),
            contracts=tuple(
                (d / name, contract) for d in run_dirs
                for name, contract in (("audit_report.json", "audit_report"),
                                       ("audit_certificate.json", "audit_certificate"),
                                       ("metrics.json", "run_metrics"))
            ),
            env=_env(ctx),
            needs=("utility.detect_poisoning",),
        ))
        stages.append(Stage(
            f"privacy.attack.seed{seed}", "privacy",
            f"Membership inference attack against every model, seed {seed}",
            module="scripts.run_mia_study",
            args=(*shared, "--seed", str(seed), "--config", *p.configs,
                  *opt("--shadow-epochs", p.shadow_epochs), *opt("--attack-epochs", p.attack_epochs)),
            inputs=tuple(d / "model.pt" for d in run_dirs),
            outputs=tuple(d / "mia_results.json" for d in run_dirs),
            contracts=tuple((d / "mia_results.json", "mia_results") for d in run_dirs),
            env=_env(ctx),
            needs=(f"privacy.train.seed{seed}",),
        ))

    stages.append(Stage(
        "privacy.report", "privacy",
        "Aggregate every seed into summary.json and figures",
        module="scripts.build_report",
        inputs=tuple(d / f for d in ctx.every_run for f in ("metrics.json", "mia_results.json")),
        outputs=(ws.runs / "summary.json",),
        contracts=((ws.runs / "summary.json", "privacy_summary"),),
        env=_env(ctx),
        needs=tuple(f"privacy.attack.seed{s}" for s in p.seeds),
        gate=seeds_gate(p.seeds),
    ))
    return stages


def certificate(ctx: Context) -> list[Stage]:
    ws = ctx.ws
    return [
        Stage(
            "privacy.certificate", "privacy", "Joint certificate: privacy evidence plus utility results",
            module="scripts.build_certificate",
            args=("--utility-results", str(ws.utility_results / "utility_results.json")),
            inputs=(ws.runs / "summary.json", ws.utility_results / "utility_results.json"),
            outputs=(
                ws.privacy / "privacy_certificate.json",
                ws.privacy / "privacy_certificate_signature.json",
            ),
            contracts=((ws.privacy / "privacy_certificate.json", "privacy_certificate"),
                       (ws.privacy / "privacy_certificate_signature.json", "audit_certificate")),
            env=_env(ctx),
            needs=("privacy.report", "utility.report"),
        ),
    ]
