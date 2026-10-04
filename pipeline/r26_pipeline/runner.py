"""Executes stages in order, in each component's own environment, and records lineage."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from r26_contracts import ContractError, validate_file
from r26_pipeline import report
from r26_pipeline.config import PipelineConfig
from r26_pipeline.stages import RunWorkspace, Stage, build_stages
from r26_pipeline.state import RunState, StageRecord, code_fingerprint, combined, fingerprint, now

COMPONENT_WORKDIRS = {"privacy": "privacy", "utility": "utility", "compliance": "compliance", "pipeline": ""}


class StageFailed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Status:
    label: str
    detail: str = ""


class Pipeline:
    def __init__(self, cfg: PipelineConfig, stages: list[Stage] | None = None, echo: bool = True) -> None:
        self.cfg = cfg
        self.ws = RunWorkspace(cfg.workspace)
        self.stages = stages if stages is not None else build_stages(cfg)
        self.by_name = {s.name: s for s in self.stages}
        self.state = RunState(self.ws.state_file)
        self.echo = echo
        self.skips = self._resolve_skips()
        self._code: dict[str, str] = {}

    def code_version(self, component: str) -> str:
        if component not in self._code:
            self._code[component] = (
                code_fingerprint(Path(__file__).resolve().parent) if component == "pipeline"
                else code_fingerprint(self.cfg.component_dir(component))
            )
        return self._code[component]

    def _resolve_skips(self) -> dict[str, str]:
        skips: dict[str, str] = {}
        for stage in self.stages:
            if stage.skip_reason:
                skips[stage.name] = stage.skip_reason
                continue
            blocked = [n for n in stage.needs if n in skips]
            if blocked:
                skips[stage.name] = f"needs {blocked[0]}, which is skipped"
        return skips

    def select(self, only: list[str] | None = None, start: str | None = None,
               until: str | None = None) -> list[Stage]:
        def index(selector: str) -> int:
            for i, stage in enumerate(self.stages):
                if stage.matches(selector):
                    return i
            raise SystemExit(f"no stage matches '{selector}'. Run 'r26 plan' to list them.")

        chosen = self.stages
        if start:
            chosen = chosen[index(start):]
        if until:
            last = max(i for i, s in enumerate(self.stages) if s.matches(until))
            chosen = [s for s in chosen if self.stages.index(s) <= last]
        if only:
            for selector in only:
                index(selector)
            chosen = [s for s in chosen if any(s.matches(sel) for sel in only)]
        return chosen

    def fingerprint(self, stage: Stage) -> str:
        parts = [f"code={self.code_version(stage.component)}", stage.module or stage.action or "",
                 *stage.args, *(f"{k}={v}" for k, v in stage.env)]
        parts += [f"{p}={fingerprint(p)}" for p in stage.inputs]
        return combined(parts)

    def status(self, stage: Stage) -> Status:
        if stage.name in self.skips:
            return Status("skip", self.skips[stage.name])
        record = self.state.get(stage.name)
        if record is None:
            return Status("pending")
        if record.status == "failed":
            return Status("failed", record.detail or "")
        if record.status != "done":
            return Status("pending")
        if record.fingerprint != self.fingerprint(stage):
            return Status("stale", "inputs or settings changed")
        for path, digest in record.outputs.items():
            if fingerprint(Path(path)) != digest:
                return Status("stale", f"{Path(path).name} changed since it was produced")
        return Status("done")

    def forecast(self, selection: list[Stage]) -> dict[str, Status]:
        """Status of each stage as a run would find it, including downstream effects."""
        statuses: dict[str, Status] = {}
        rerun_outputs: list[Path] = []
        rerunning: set[str] = set()
        for stage in self.stages:
            status = self.status(stage)
            if status.label == "done":
                upstream = next((n for n in stage.needs if n in rerunning), None)
                touched = next((p for p in stage.inputs for o in rerun_outputs
                                if p == o or o in p.parents or p in o.parents), None)
                if upstream or touched:
                    status = Status("stale", f"upstream {upstream or touched.name} will rerun")
            if status.label in ("pending", "stale", "failed"):
                rerunning.add(stage.name)
                rerun_outputs.extend(stage.outputs)
            statuses[stage.name] = status
        return {s.name: statuses[s.name] for s in selection}

    def _say(self, message: str) -> None:
        if self.echo:
            print(message, flush=True)

    def _command(self, stage: Stage) -> list[str]:
        return self.cfg.environments[stage.component].command("-m", stage.module, *stage.args)

    def _child_env(self, stage: Stage) -> dict[str, str]:
        env = os.environ.copy()
        component_dir = str(self.cfg.component_dir(stage.component))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [component_dir, env.get("PYTHONPATH")]))
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        env.update(dict(stage.env))
        return env

    def _execute(self, stage: Stage, log_path: Path) -> int:
        workdir = self.ws.root / COMPONENT_WORKDIRS[stage.component]
        workdir.mkdir(parents=True, exist_ok=True)
        command = self._command(stage)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# {stage.name}  started {now()}\n# cwd {workdir}\n# {' '.join(command)}\n\n")
            log.flush()
            process = subprocess.Popen(
                command, cwd=workdir, env=self._child_env(stage), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    if self.echo:
                        sys.stdout.write("    " + line)
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise

    def _run_internal(self, stage: Stage) -> None:
        if stage.action == "release":
            copied = report.assemble_release(self.ws)
            self._say(f"    released {len(copied)} files to {self.ws.release}")
        elif stage.action == "final":
            result = report.final_report(self.ws, self.stages, self.state)
            if self.echo:
                report.print_summary(result)
        else:
            raise StageFailed(f"unknown internal action '{stage.action}'")

    def _check_ready(self, stage: Stage) -> None:
        for need in stage.needs:
            record = self.state.get(need)
            if record is None or record.status != "done":
                raise StageFailed(f"needs {need}, which has not completed. Run it first.")

    def run_stage(self, stage: Stage) -> StageRecord:
        log_path = self.ws.logs / f"{stage.name}.log"
        started, clock = now(), time.monotonic()
        record = StageRecord(status="running", started=started, log=str(log_path))
        self.state.set(stage.name, record)

        try:
            self._check_ready(stage)
            for path in stage.clean:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
            if stage.internal:
                self._run_internal(stage)
                record.exit_code = 0
            else:
                record.exit_code = self._execute(stage, log_path)
                if record.exit_code != 0:
                    raise StageFailed(f"exited with code {record.exit_code}; see {log_path}")

            missing = [p for p in stage.outputs if not p.exists()]
            if missing:
                raise StageFailed(f"did not produce {', '.join(str(p) for p in missing)}")
            for path, contract in stage.contracts:
                validate_file(path, contract)
            if stage.gate is not None:
                problem = stage.gate(self.ws)
                if problem:
                    raise StageFailed(f"gate failed: {problem}")
        except (StageFailed, ContractError) as exc:
            record.status, record.detail = "failed", str(exc)
        except KeyboardInterrupt:
            record.status, record.detail = "failed", "interrupted"
            self.state.set(stage.name, record)
            raise
        else:
            record.status = "done"
            record.code = self.code_version(stage.component)
            record.fingerprint = self.fingerprint(stage)
            record.outputs = {str(p): fingerprint(p) for p in stage.outputs}
        finally:
            record.finished = now()
            record.seconds = round(time.monotonic() - clock, 1)
            self.state.set(stage.name, record)
        return record

    def run(self, selection: list[Stage], force: bool = False, dry_run: bool = False) -> int:
        self.ws.root.mkdir(parents=True, exist_ok=True)
        forecast = self.forecast(selection) if dry_run else {}
        for position, stage in enumerate(selection, start=1):
            header = f"[{position}/{len(selection)}] {stage.name}"
            current = forecast.get(stage.name) or self.status(stage)
            if current.label == "skip":
                self._say(f"{header}  skipped: {current.detail}")
                if not dry_run:
                    skipped = StageRecord(status="skipped", detail=current.detail, finished=now())
                    self.state.set(stage.name, skipped)
                continue
            if current.label == "done" and not force:
                self._say(f"{header}  up to date")
                continue
            if dry_run:
                where = self.cfg.environments.get(stage.component)
                command = "(inside the pipeline)" if stage.internal else " ".join(self._command(stage))
                self._say(f"{header}  would run [{where.describe() if where else 'pipeline'}]\n    {command}")
                continue

            self._say(f"{header}  {stage.summary}")
            record = self.run_stage(stage)
            if record.status != "done":
                self._say(f"{header}  FAILED after {record.seconds}s: {record.detail}")
                return 1
            self._say(f"{header}  done in {record.seconds}s")
        return 0
