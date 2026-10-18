"""A printable, one-page PDF of privacy_certificate.json.

The PDF is a readable summary; its trust comes from the fingerprint. It prints
the SHA-256 of the exact bytes of privacy_certificate.json and the HMAC-SHA256
signature from privacy_certificate_signature.json, stores the hash in the PDF
metadata and a QR code, and writes a record with the SHA-256 of the PDF
itself, so `verify_pdf` shows that neither file changed. It matches the
Risk and Compliance certificate in layout. It is a project record of the
privacy guarantees enforced and tested, not a legal certification, and its
signature lines are left blank to be signed by hand.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT = "R26-CS-010"
PROJECT_TITLE = "Privacy-Enhanced Synthetic Data Generation for Autonomous Vehicle Training"
INSTITUTION = "Sri Lanka Institute of Information Technology"
CONFIG_LABELS = {"no_dp": "No DP (control)", "config_a": "Strong", "config_b": "Medium", "config_c": "Weak"}

SIGNATORIES: tuple[tuple[str, str, str], ...] = (
    ("Issued by", "Priyadarshi S.A.S.D. (IT22309556)", "Privacy Integration"),
    ("Verified by", "Wijesinghe R.D.P.T. (IT22066916)", "Risk and Compliance"),
    ("Reviewed by", "Prof. Anuradha Jayakody", "Supervisor"),
)

INK = (0.08, 0.13, 0.24)
MUTED = (0.33, 0.38, 0.45)
RULE = (0.80, 0.84, 0.88)
PAPER = (0.985, 0.983, 0.975)
VERDICT_COLORS = {"PASS": (0.11, 0.42, 0.31), "FAIL": (0.64, 0.17, 0.10)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certificate_id(digest: str) -> str:
    return f"R26-PI-{digest[:12].upper()}"


def _fmt(value, digits: int = 3) -> str:
    """Round half up, as the reports do (0.6385 -> 0.639), not to the nearest binary float."""
    if value is None:
        return "n/a"
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


@dataclass(slots=True)
class PdfData:
    verdict: str
    enforcement: str
    validation: str
    digest: str
    signature: str | None
    issued_at: str
    rows: list[tuple[str, str, str, str, str, str]]
    lines: list[str]
    limitations: list[str] = field(default_factory=list)

    @property
    def cert_id(self) -> str:
        return certificate_id(self.digest)


def collect(cert_path: Path, signature_path: Path | None, issued_at: str | None = None) -> PdfData:
    cert = json.loads(Path(cert_path).read_text(encoding="utf-8"))
    digest = sha256_file(cert_path)
    signature = None
    if signature_path is not None and Path(signature_path).is_file():
        record = json.loads(Path(signature_path).read_text(encoding="utf-8"))
        if record.get("sha256_content") == digest:
            signature = record.get("hmac_signature")

    configs = cert.get("configurations", {})
    order = sorted(configs, key=lambda t: (t != "no_dp", -(configs[t].get("target_epsilon") or 0)))
    rows = []
    for tag in order:
        c = configs[tag]
        eps = c.get("target_epsilon")
        rows.append(
            (
                CONFIG_LABELS.get(tag, tag),
                "none" if eps is None else f"{eps:g}",
                _fmt((c.get("achieved_epsilon") or {}).get("mean"), 2) if eps is not None else "none",
                _fmt((c.get("utility_auc") or {}).get("mean")),
                _fmt((c.get("attack_auc") or {}).get("mean")),
                str(c.get("replications", "")),
            )
        )

    mech = cert.get("privacy_mechanism", {})
    cal, train = mech.get("calibration_stage", {}), mech.get("training_stage", {})
    lines = []
    if cal:
        lines.append(
            f"Calibration: Laplace on {cal.get('queries')} statistics, epsilon {cal.get('epsilon_total')} in "
            f"total ({cal.get('epsilon_per_query')} per query, sequential composition)."
        )
    if train:
        lines.append(
            f"Training: DP-SGD (Opacus, RDP accountant), clipping C = {train.get('clipping_norm_C')}, "
            f"delta = {train.get('delta')}, hard ceiling epsilon {train.get('epsilon_ceiling')}."
        )
    stress = cert.get("stress_test", {})
    if stress.get("status") == "run" and stress.get("verdict"):
        comps = stress.get("comparisons") or []
        seeds = len((stress.get("regime") or {}).get("seeds") or [])
        worst = max((c.get("p_value_one_sided") or 0 for c in comps), default=None)
        base = comps[0]["no_dp_attack_auc_mean"] if comps else None
        best = min((c["dp_attack_auc_mean"] for c in comps), default=None)
        lines.append(
            f"Stress test ({seeds} seeds, overfitted model): attack AUC {_fmt(base)} without DP, "
            f"{_fmt(best)} at the strongest budget; reduction {stress['verdict'].get('dp_reduction')} "
            f"(p <= {_fmt(worst, 4)})."
        )
    calibrated = cert.get("calibrated_attack", {})
    if calibrated.get("status") == "run":
        low = {c["config"]: c for c in calibrated.get("low_fpr_comparisons") or []}
        strongest = min(low.values(), key=lambda c: c["dp_tpr_at_1pct_fpr_mean"], default=None)
        if strongest:
            lines.append(
                "Calibrated attack (after LiRA): members caught at 1% FPR fell from "
                f"{strongest['no_dp_tpr_at_1pct_fpr_mean'] * 100:.1f}% without DP to "
                f"{strongest['dp_tpr_at_1pct_fpr_mean'] * 100:.1f}% at "
                f"{strongest['config'].replace('eps_', 'epsilon ')}; chance is 1%."
            )
    limitations = [
        f"[{d.get('severity', '')}]  {d.get('issue', '')}" for d in cert.get("declared_limitations", [])
    ]
    verdicts = cert.get("certification", {})
    return PdfData(
        verdict=str(verdicts.get("overall_privacy_verdict", "FAIL")),
        enforcement=str(verdicts.get("privacy_enforcement", "n/a")),
        validation=str(verdicts.get("empirical_validation", "n/a")),
        digest=digest,
        signature=signature,
        issued_at=issued_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
        rows=rows,
        lines=lines,
        limitations=limitations,
    )


def _wrap(text: str, font: str, size: float, width: float, c) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _seal(c, cx: float, cy: float, radius: float, color, verdict: str) -> None:
    c.saveState()
    c.setStrokeColorRGB(*color)
    c.setFillColorRGB(*color)
    c.setLineWidth(2.2)
    c.circle(cx, cy, radius, stroke=1, fill=0)
    c.setLineWidth(0.8)
    c.circle(cx, cy, radius - 6, stroke=1, fill=0)
    c.circle(cx, cy, radius - 25, stroke=1, fill=0)
    for i in range(72):
        a = 2 * math.pi * i / 72
        r1, r2 = radius - 6, radius - 3
        c.line(cx + r1 * math.cos(a), cy + r1 * math.sin(a), cx + r2 * math.cos(a), cy + r2 * math.sin(a))
    text = f"{PROJECT}  *  PRIVACY INTEGRATION  *  VERIFICATION RECORD  *  "
    size = 8.2
    c.setFont("Helvetica-Bold", size)
    arc_r = radius - 16
    widths = [c.stringWidth(ch, "Helvetica-Bold", size) for ch in text]
    stretch = (2 * math.pi * arc_r) / sum(widths)
    angle = math.pi / 2 + (widths[0] * stretch / 2) / arc_r
    for ch, w in zip(text, widths, strict=True):
        step = w * stretch / arc_r
        angle -= step / 2
        c.saveState()
        c.translate(cx + arc_r * math.cos(angle), cy + arc_r * math.sin(angle))
        c.rotate(math.degrees(angle) - 90)
        c.drawCentredString(0, -3, ch)
        c.restoreState()
        angle -= step / 2
    c.setFont("Helvetica-Bold", 17)
    c.drawCentredString(cx, cy - 2, verdict)
    c.setFont("Helvetica", 5.8)
    c.drawCentredString(cx, cy - 13, "PRIVACY VERDICT")
    c.setFont("Helvetica", 7)
    c.drawCentredString(cx, cy + 12, PROJECT)
    c.restoreState()


def render(data: PdfData, output: Path) -> None:
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    c = pdfcanvas.Canvas(str(output), pagesize=A4, invariant=1)
    c.setTitle(f"{PROJECT} Privacy Certificate {data.cert_id}")
    c.setAuthor("R26-CS-010 Privacy Integration (IT22309556)")
    c.setSubject(f"sha256:{data.digest}")
    c.setKeywords([PROJECT, "privacy", f"sha256:{data.digest}", data.verdict])
    color = VERDICT_COLORS.get(data.verdict, VERDICT_COLORS["FAIL"])
    m = 46

    c.setFillColorRGB(*PAPER)
    c.rect(0, 0, width, height, stroke=0, fill=1)
    c.setStrokeColorRGB(*INK)
    c.setLineWidth(1.6)
    c.rect(22, 22, width - 44, height - 44, stroke=1, fill=0)
    c.setLineWidth(0.5)
    c.rect(28, 28, width - 56, height - 56, stroke=1, fill=0)
    c.setFillColorRGB(*color)
    c.rect(28, height - 34, width - 56, 6, stroke=0, fill=1)

    y = height - 74
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, y, f"{PROJECT}  |  {INSTITUTION.upper()}")
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(width - m, y, f"Certificate ID  {data.cert_id}")

    y -= 52
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 40)
    c.drawString(m, y, "Privacy")
    y -= 44
    c.drawString(m, y, "Certificate")
    y -= 30
    intro = (
        f"This certificate records the differential privacy guarantees enforced and tested by the "
        f"Privacy Integration component of {PROJECT}, {PROJECT_TITLE}. It summarises "
        "privacy_certificate.json; the SHA-256 fingerprint below identifies that file exactly, and "
        "Risk and Compliance verifies its signature independently."
    )
    c.setFont("Helvetica", 10.5)
    for line in _wrap(intro, "Helvetica", 10.5, width - 2 * m - 150, c):
        c.drawString(m, y, line)
        y -= 15
    _seal(c, width - m - 62, height - 176, 60, color, data.verdict)

    y -= 10
    c.setFillColorRGB(*color)
    c.roundRect(m, y - 34, width - 2 * m, 34, 6, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(m + 14, y - 22, f"OVERALL PRIVACY VERDICT: {data.verdict}")
    c.setFont("Helvetica", 9.5)
    c.drawRightString(
        width - m - 14, y - 21, f"Enforcement {data.enforcement}  |  Empirical validation {data.validation}"
    )
    y -= 54

    # Results table.
    # Configuration names are left-aligned; every result column is centred on its header.
    left = m + 8
    centres = [m + 170, m + 252, m + 330, m + 402, m + 470]
    heads = ["Configuration", "Target epsilon", "Achieved epsilon", "Test AUC", "Attack AUC", "Seeds"]
    c.setFillColorRGB(*INK)
    c.rect(m, y - 18, width - 2 * m, 18, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(left, y - 13, heads[0])
    for centre, h in zip(centres, heads[1:], strict=True):
        c.drawCentredString(centre, y - 13, h)
    y -= 18
    c.setStrokeColorRGB(*RULE)
    for row in data.rows:
        y -= 16
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold" if row[0].startswith("No DP") else "Helvetica", 9.5)
        c.drawString(left, y + 4, row[0])
        for centre, cell in zip(centres, row[1:], strict=True):
            c.drawCentredString(centre, y + 4, cell)
        c.line(m, y, width - m, y)
    y -= 18

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(*INK)
    for text in data.lines:
        for i, line in enumerate(_wrap(text, "Helvetica", 9, width - 2 * m - 12, c)):
            c.drawString(m + (4 if i == 0 else 12), y, ("- " if i == 0 else "") + line)
            y -= 12
        y -= 2
    if data.limitations:
        y -= 4
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(m, y, "Declared limitations")
        y -= 13
        c.setFont("Helvetica", 8.8)
        c.setFillColorRGB(*MUTED)
        for item in data.limitations[:5]:
            c.drawString(m + 4, y, item[:115])
            y -= 11.5
    y -= 4

    box_h = 104
    c.setFillColorRGB(0.94, 0.95, 0.96)
    c.setStrokeColorRGB(*RULE)
    c.roundRect(m, y - box_h, width - 2 * m, box_h, 6, stroke=1, fill=1)
    qr = 86
    widget = QrCodeWidget(f"{PROJECT}|privacy_certificate.json|sha256:{data.digest}")
    x0, y0, x1, y1 = widget.getBounds()
    drawing = Drawing(qr, qr, transform=[qr / (x1 - x0), 0, 0, qr / (y1 - y0), 0, 0])
    drawing.add(widget)
    renderPDF.draw(drawing, c, width - m - qr - 9, y - box_h + 9)
    tx, ty = m + 14, y - 18
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(tx, ty, "SHA-256 of privacy_certificate.json")
    c.setFont("Courier", 9.5)
    c.drawString(tx, ty - 13, data.digest[:32])
    c.drawString(tx, ty - 25, data.digest[32:])
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(tx, ty - 43, "HMAC-SHA256 signature (privacy_certificate_signature.json)")
    c.setFont("Courier", 9.5)
    if data.signature:
        c.drawString(tx, ty - 56, data.signature[:32])
        c.drawString(tx, ty - 68, data.signature[32:])
    else:
        c.drawString(tx, ty - 56, "not signed, or the signature does not match this file")
    c.setFont("Helvetica", 8.5)
    c.setFillColorRGB(*MUTED)
    c.drawString(tx, ty - 80, f"Issued {data.issued_at.replace('T', ' ').replace('+00:00', ' UTC')}.")
    y -= box_h + 12

    col_w = (width - 2 * m - 36) / 3
    for i, (role, name, title) in enumerate(SIGNATORIES):
        x = m + i * (col_w + 18)
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica-Oblique", 8.5)
        c.drawString(x, y, role)
        c.setStrokeColorRGB(*INK)
        c.setLineWidth(0.7)
        c.line(x, y - 28, x + col_w, y - 28)
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x, y - 41, name)
        c.setFont("Helvetica", 8.5)
        c.setFillColorRGB(*MUTED)
        c.drawString(x, y - 53, f"{title}    Date: ____________")

    c.setFont("Helvetica", 7.8)
    c.setFillColorRGB(*MUTED)
    note = (
        "Verify: python -m scripts.build_certificate --verify. A project record of the privacy "
        "guarantees enforced and tested, not a legal certification of GDPR compliance."
    )
    fy = 44
    for line in reversed(_wrap(note, "Helvetica", 7.8, width - 2 * m, c)):
        c.drawString(m, fy, line)
        fy += 10
    c.showPage()
    c.save()


def record(data: PdfData, pdf_path: Path) -> dict:
    return {
        "project": PROJECT,
        "certificate_id": data.cert_id,
        "source": "privacy_certificate.json",
        "overall_privacy_verdict": data.verdict,
        "sha256_certificate": data.digest,
        "hmac_signature": data.signature,
        "sha256_pdf": sha256_file(pdf_path),
        "issued_at": data.issued_at,
    }


@dataclass(frozen=True, slots=True)
class PdfCheck:
    certificate_matches: bool
    pdf_matches: bool
    pdf_embeds_hash: bool

    @property
    def passed(self) -> bool:
        return self.certificate_matches and self.pdf_matches and self.pdf_embeds_hash


def verify_pdf(cert_path: Path, pdf_path: Path, record_path: Path) -> PdfCheck:
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    digest = sha256_file(cert_path)
    return PdfCheck(
        certificate_matches=hmac.compare_digest(digest, str(rec.get("sha256_certificate", ""))),
        pdf_matches=hmac.compare_digest(sha256_file(pdf_path), str(rec.get("sha256_pdf", ""))),
        pdf_embeds_hash=f"sha256:{digest}".encode("ascii") in Path(pdf_path).read_bytes(),
    )
