"""r26: run the integrated R26-CS-010 pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from r26_common.env import load_env

from r26_contracts import ContractError, validate_file
from r26_pipeline.config import (
    COMPONENT_FOLDERS,
    COMPONENTS,
    REQUIRED_SECRETS,
    ConfigError,
    load_config,
)
from r26_pipeline.runner import Pipeline

IMPORT_CHECKS = {
    "privacy": "import privacy_integration, opacus",
    "utility": "import utility_evaluation, ultralytics",
    "compliance": "import risk_compliance",
}


def _default_config() -> Path:
    here = Path.cwd() / "pipeline.yaml"
    return here if here.is_file() else Path(__file__).resolve().parents[1] / "pipeline.yaml"


def _pipeline(args: argparse.Namespace, echo: bool = True) -> Pipeline:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from None
    load_env(cfg.repository / ".env")
    return Pipeline(cfg, echo=echo)


def cmd_plan(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, echo=False)
    print(f"workspace  {pipeline.ws.root}\n")
    print(f"{'stage':<30}{'component':<12}{'status':<12}detail")
    print("-" * 88)
    selection = pipeline.select(args.only, args.start, args.until)
    forecast = pipeline.forecast(selection)
    for stage in selection:
        status = forecast[stage.name]
        print(f"{stage.name:<30}{stage.component:<12}{status.label:<12}{status.detail or stage.summary}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    selection = pipeline.select(args.only, args.start, args.until)
    if not args.dry_run:
        missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name)]
        if missing:
            raise SystemExit(f"missing secrets: {', '.join(missing)}. Add them to .env at the repo root.")
    return pipeline.run(selection, force=args.force, dry_run=args.dry_run)


def cmd_status(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, echo=False)
    print(f"{'stage':<30}{'state':<10}{'finished':<27}{'seconds':>9}  detail")
    print("-" * 96)
    for stage in pipeline.stages:
        record = pipeline.state.get(stage.name)
        if record is None:
            print(f"{stage.name:<30}{'not run':<10}")
            continue
        seconds = "" if record.seconds is None else f"{record.seconds:.0f}"
        finished = record.finished or ""
        print(f"{stage.name:<30}{record.status:<10}{finished:<27}{seconds:>9}  {record.detail or ''}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, echo=False)
    cfg = pipeline.cfg
    problems = 0

    def line(ok: bool, label: str, detail: str) -> None:
        nonlocal problems
        problems += not ok
        print(f"  {'ok ' if ok else 'NO '} {label:<26}{detail}")

    print("repository")
    for component in COMPONENTS:
        folder = cfg.component_dir(component)
        line(folder.is_dir(), COMPONENT_FOLDERS[component], str(folder))

    print("environments")
    for component in COMPONENTS:
        env = cfg.environments[component]
        try:
            result = subprocess.run(
                env.command("-c", IMPORT_CHECKS[component]), cwd=cfg.component_dir(component),
                capture_output=True, text=True, timeout=300, check=False,
            )
            ok, detail = result.returncode == 0, env.describe()
            if not ok:
                detail += f": {(result.stderr.strip().splitlines() or ['import failed'])[-1]}"
        except (OSError, subprocess.SubprocessError, ConfigError) as exc:
            ok, detail = False, f"{env.describe()}: {exc}"
        line(ok, component, detail)

    print("inputs")
    line(cfg.inputs.encrypted_dataset.is_dir(), "encrypted dataset", str(cfg.inputs.encrypted_dataset))
    for label, path in (("calibration source", cfg.inputs.calibration_source), ("KITTI", cfg.inputs.kitti)):
        if path is None:
            print(f"  --  {label:<26}not set; dependent stages will be skipped")
        else:
            line(path.exists(), label, str(path))

    print("secrets")
    for name in REQUIRED_SECRETS:
        line(bool(os.environ.get(name)), name, "set" if os.environ.get(name) else "missing")

    print(f"\n{'ready' if not problems else f'{problems} problem(s) to fix'}")
    return 1 if problems else 0


def cmd_validate(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args, echo=False)
    checked = failed = 0
    for stage in pipeline.stages:
        for path, contract in stage.contracts:
            if not path.is_file():
                continue
            checked += 1
            try:
                validate_file(path, contract)
                print(f"  ok   {contract:<24}{path.relative_to(pipeline.ws.root)}")
            except ContractError as exc:
                failed += 1
                print(f"  FAIL {contract:<24}{exc}")
    print(f"\n{checked} handoff files checked, {failed} failed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="r26", description="Integrated R26-CS-010 pipeline")
    parser.add_argument("--config", type=Path, default=None, help="pipeline.yaml (default: ./pipeline.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--only", nargs="+", help="stage names or prefixes, e.g. privacy or utility.fid")
        p.add_argument("--from", dest="start", help="start at this stage")
        p.add_argument("--until", help="stop after this stage")

    plan = sub.add_parser("plan", help="list stages and whether each will run")
    selection(plan)
    run = sub.add_parser("run", help="run the pipeline")
    selection(run)
    run.add_argument("--force", action="store_true", help="rerun stages even if up to date")
    run.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    sub.add_parser("status", help="results of the last run")
    sub.add_parser("check", help="check environments, inputs and secrets")
    sub.add_parser("validate", help="validate every handoff file against the shared contracts")

    args = parser.parse_args(argv)
    args.config = args.config or _default_config()
    handlers = {"plan": cmd_plan, "run": cmd_run, "status": cmd_status, "check": cmd_check,
                "validate": cmd_validate}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
