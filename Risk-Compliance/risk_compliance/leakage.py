"""Metadata leakage scan."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "absolute_windows_path": re.compile(r"[A-Za-z]:\\\\(?:[^\"\\\\]+\\\\)*"),
    "absolute_unix_path": re.compile(r"/(?:home|Users|root|tmp|mnt|media|opt|var)/[^\"\s]+"),
    "email_address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "ip_address": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    # Same line only, so lists of decimals in JSON are not read as coordinates.
    "gps_coordinate": re.compile(r"\b-?\d{1,3}\.\d{4,}[ \t]*,[ \t]*-?\d{1,3}\.\d{4,}\b"),
    "possible_credential": re.compile(
        r"(?i)(secret|api[_-]?key|password|token)\s*[=:]\s*\S+"
    ),
    "machine_identifier": re.compile(r"(?i)\b(hostname|computername|username)\b"),
}


@dataclass(frozen=True, slots=True)
class Finding:
    file: str
    issue: str
    sample: str

    def __str__(self) -> str:
        return f"{self.file}  {self.issue}  {self.sample}"


@dataclass
class LeakageResult:
    files_scanned: int
    findings: list[Finding]

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "REVIEW REQUIRED"

    def by_issue(self) -> dict[str, list[Finding]]:
        grouped: dict[str, list[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.issue, []).append(finding)
        return grouped

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "files_scanned": self.files_scanned,
            "finding_count": len(self.findings),
            "by_issue": {
                issue: len(items) for issue, items in sorted(self.by_issue().items())
            },
            "findings": [str(f) for f in self.findings[:100]],
        }


def scan(root: Path, suffixes: frozenset[str], sample_length: int = 80) -> LeakageResult:
    findings: list[Finding] = []
    scanned = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        relative = str(path.relative_to(root))
        for issue, pattern in PATTERNS.items():
            for match in pattern.findall(text):
                sample = match if isinstance(match, str) else " ".join(match)
                findings.append(Finding(relative, issue, sample[:sample_length]))

    return LeakageResult(files_scanned=scanned, findings=findings)
