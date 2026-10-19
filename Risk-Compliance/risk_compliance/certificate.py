"""The compliance certificate: a signed, printable PDF of compliance_report.json.

The PDF is a readable summary. What makes it trustworthy is the fingerprint:
the SHA-256 of the exact bytes of compliance_report.json, printed on the page,
stored in the PDF's metadata, encoded in the QR code and signed with
HMAC-SHA256. A sidecar JSON records the same values plus the SHA-256 of the
PDF itself, so `verify_certificate` can show that neither file was changed.

The certificate is a project verification record. It is not a legal
certification of GDPR compliance. It has one signature line, left blank to
be signed by hand by the component lead.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ALGORITHM = "HMAC-SHA256"
PROJECT = "R26-CS-010"
PROJECT_TITLE = "Privacy-Enhanced Synthetic Data Generation for Autonomous Vehicle Training"
INSTITUTION = "Sri Lanka Institute of Information Technology"

CHECK_LABELS: dict[str, str] = {
    "certificate_verification": "Signed audit certificates",
    "budget_audit": "Privacy budget logs",
    "calibration_audit_log": "Calibration audit log reconciliation",
    "calibration_composition": "Calibration budget composition",
    "k_anonymity": "k-anonymity of released metadata",
    "manifest_verification": "Dataset manifest (SHA-256)",
    "poisoning_detection": "Data poisoning detection",
}

SIGNATORY: tuple[str, str, str] = (
    "Wijesinghe R.D.P.T.",
    "Component Lead, Risk and Compliance (IT22066916)",
    f"{PROJECT} Risk and Compliance",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign(digest: str, key: bytes) -> str:
    return hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()


def certificate_id(digest: str) -> str:
    return f"R26-RC-{digest[:12].upper()}"


def qr_payload(digest: str) -> str:
    return f"{PROJECT}|compliance_report.json|sha256:{digest}"


def _check_rows(report: dict) -> list[tuple[str, str]]:
    run = set(report.get("checks_run", []))
    mapping_verified: dict[str, bool | None] = {}
    for entry in report.get("regulatory_mapping", []):
        mapping_verified.setdefault(entry.get("evidence_check"), entry.get("verified"))
    rows = []
    for key, label in CHECK_LABELS.items():
        if key not in run and key not in report.get("checks_not_run", []):
            continue
        if key not in run:
            state = "Not run"
        else:
            verified = mapping_verified.get(key)
            state = {True: "Passed", False: "Failed", None: "Recorded"}[verified]
        rows.append((label, state))
    return rows


@dataclass(slots=True)
class CertificateData:
    verdict: str
    digest: str
    signature: str | None
    issued_at: str
    report_generated_at: str | None
    checks: list[tuple[str, str]]
    controls_verified: int
    controls_total: int
    weaknesses: list[dict] = field(default_factory=list)

    @property
    def cert_id(self) -> str:
        return certificate_id(self.digest)


def collect(report_path: Path, key: bytes | None, issued_at: str | None = None) -> CertificateData:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    digest = sha256_file(report_path)
    mapping = report.get("regulatory_mapping", [])
    return CertificateData(
        verdict=str(report.get("overall_verdict", "INCOMPLETE")),
        digest=digest,
        signature=sign(digest, key) if key else None,
        issued_at=issued_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
        report_generated_at=report.get("generated_at"),
        checks=_check_rows(report),
        controls_verified=sum(1 for m in mapping if m.get("verified") is True),
        controls_total=len(mapping),
        weaknesses=list(report.get("declared_weaknesses", [])),
    )


# ---------- Rendering ----------

INK = (0.08, 0.13, 0.24)
MUTED = (0.33, 0.38, 0.45)
RULE = (0.80, 0.84, 0.88)
PAPER = (0.985, 0.983, 0.975)
VERDICT_COLORS = {
    "PASS": (0.11, 0.42, 0.31),
    "INCOMPLETE": (0.72, 0.47, 0.12),
    "FAIL": (0.64, 0.17, 0.10),
}


def _wrap(text: str, font: str, size: float, width: float, canvas) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if canvas.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _seal(c, cx: float, cy: float, radius: float, color: tuple[float, float, float], verdict: str) -> None:
    """The project's own seal: two rings, curved text and the verdict in the middle."""
    c.saveState()
    c.setStrokeColorRGB(*color)
    c.setFillColorRGB(*color)
    c.setLineWidth(2.2)
    c.circle(cx, cy, radius, stroke=1, fill=0)
    c.setLineWidth(0.8)
    c.circle(cx, cy, radius - 6, stroke=1, fill=0)
    c.circle(cx, cy, radius - 25, stroke=1, fill=0)
    # Small ticks between the rings, like a stamped edge.
    for i in range(72):
        a = 2 * math.pi * i / 72
        r1, r2 = radius - 6, radius - 3
        c.line(cx + r1 * math.cos(a), cy + r1 * math.sin(a), cx + r2 * math.cos(a), cy + r2 * math.sin(a))
    text = f"{PROJECT}  *  RISK AND COMPLIANCE  *  VERIFICATION RECORD  *  "
    size = 8.2
    c.setFont("Helvetica-Bold", size)
    arc_r = radius - 16
    widths = [c.stringWidth(ch, "Helvetica-Bold", size) for ch in text]
    stretch = (2 * math.pi * arc_r) / sum(widths)  # spread the text evenly around the whole ring
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
    c.drawCentredString(cx, cy - 13, "OVERALL VERDICT")
    c.drawCentredString(cx, cy + 12, PROJECT)
    c.restoreState()


def _signature_block(
    c,
    left: float,
    right: float,
    top: float,
    signer: tuple[str, str, str],
    issued: str,
    cert_id: str,
) -> None:
    """One company-style signature: issue details on the left, one signature line on the right.

    The line is left blank to be signed by hand by the component lead.
    """
    name, title, org = signer
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(left, top - 8, "DATE OF ISSUE")
    c.drawString(left, top - 36, "CERTIFICATE ID")
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, top - 21, issued)
    c.drawString(left, top - 49, cert_id)

    bw = 220
    bx = right - bw
    line_y = top - 40
    c.setStrokeColorRGB(*INK)
    c.setLineWidth(0.8)
    c.line(bx, line_y, right, line_y)
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(bx, line_y - 13, name)
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 8.3)
    c.drawString(bx, line_y - 24, title)
    c.drawString(bx, line_y - 34, org)


def render_pdf(data: CertificateData, output: Path) -> None:
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    c = pdfcanvas.Canvas(str(output), pagesize=A4, invariant=1)
    c.setTitle(f"{PROJECT} Privacy Compliance Certificate {data.cert_id}")
    c.setAuthor("R26-CS-010 Risk and Compliance (IT22066916)")
    c.setSubject(f"sha256:{data.digest}")
    c.setKeywords([PROJECT, "compliance", f"sha256:{data.digest}", data.verdict])
    color = VERDICT_COLORS.get(data.verdict, VERDICT_COLORS["INCOMPLETE"])
    margin = 46

    # Paper and frame.
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
    c.setFillColorRGB(*color)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, f"{PROJECT}  |  {INSTITUTION.upper()}")
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(width - margin, y, f"Certificate ID  {data.cert_id}")

    y -= 52
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 40)
    c.drawString(margin, y, "Privacy Compliance")
    y -= 44
    c.drawString(margin, y, "Certificate")

    y -= 30
    intro = (
        f"This certificate records the result of the independent compliance verification of the "
        f"{PROJECT} pipeline, {PROJECT_TITLE}, carried out by the Risk and Compliance component. "
        "It summarizes compliance_report.json. The SHA-256 fingerprint below identifies that report "
        "exactly: if a single byte of the report changes, the fingerprint no longer matches."
    )
    c.setFont("Helvetica", 10.5)
    for line in _wrap(intro, "Helvetica", 10.5, width - 2 * margin - 150, c):
        c.drawString(margin, y, line)
        y -= 15

    # Seal, right of the title block.
    _seal(c, width - margin - 62, height - 176, 60, color, data.verdict)

    # Verdict band.
    y -= 10
    c.setFillColorRGB(*color)
    c.roundRect(margin, y - 34, width - 2 * margin, 34, 6, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin + 14, y - 22, f"OVERALL VERDICT: {data.verdict}")
    c.setFont("Helvetica", 9.5)
    c.drawRightString(
        width - margin - 14,
        y - 21,
        f"{data.controls_verified} of {data.controls_total} GDPR controls verified  |  "
        f"{len(data.weaknesses)} declared weaknesses",
    )
    y -= 56

    # Checks table.
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Independent checks")
    y -= 8
    c.setStrokeColorRGB(*RULE)
    c.setLineWidth(0.6)
    for label, state in data.checks:
        c.line(margin, y, width - margin, y)
        y -= 16
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica", 10)
        c.drawString(margin + 4, y + 4, label)
        state_color = {"Passed": VERDICT_COLORS["PASS"], "Failed": VERDICT_COLORS["FAIL"]}.get(state, MUTED)
        c.setFillColorRGB(*state_color)
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(width - margin - 4, y + 4, state)
    c.line(margin, y, width - margin, y)
    y -= 22

    if data.weaknesses:
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Declared weaknesses")
        y -= 15
        c.setFont("Helvetica", 9.5)
        c.setFillColorRGB(*MUTED)
        for w in data.weaknesses[:4]:
            c.drawString(margin + 4, y, f"[{w.get('severity', '')}]  {w.get('issue', '')}"[:110])
            y -= 13
        y -= 8

    # Fingerprint block with the QR code.
    box_h = 112
    c.setFillColorRGB(0.94, 0.95, 0.96)
    c.setStrokeColorRGB(*RULE)
    c.roundRect(margin, y - box_h, width - 2 * margin, box_h, 6, stroke=1, fill=1)
    qr_size = 92
    widget = QrCodeWidget(qr_payload(data.digest))
    x0, y0, x1, y1 = widget.getBounds()
    drawing = Drawing(qr_size, qr_size, transform=[qr_size / (x1 - x0), 0, 0, qr_size / (y1 - y0), 0, 0])
    drawing.add(widget)
    renderPDF.draw(drawing, c, width - margin - qr_size - 10, y - box_h + 10)
    tx = margin + 14
    ty = y - 20
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(tx, ty, "SHA-256 of compliance_report.json")
    c.setFont("Courier", 9.5)
    c.drawString(tx, ty - 14, data.digest[:32])
    c.drawString(tx, ty - 26, data.digest[32:])
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(tx, ty - 46, f"{ALGORITHM} signature")
    c.setFont("Courier", 9.5)
    if data.signature:
        c.drawString(tx, ty - 60, data.signature[:32])
        c.drawString(tx, ty - 72, data.signature[32:])
    else:
        c.drawString(tx, ty - 60, "not signed (R26_AUDIT_HMAC_KEY not set)")
    c.setFont("Helvetica", 8.5)
    c.setFillColorRGB(*MUTED)
    issued = data.issued_at.replace("T", " ").replace("+00:00", " UTC")
    generated = (data.report_generated_at or "unknown")[:19].replace("T", " ")
    c.drawString(tx, ty - 88, f"Certificate issued {issued}.  Report generated {generated} UTC.")
    y -= box_h + 26

    # One digital signature, company style.
    _signature_block(c, margin, width - margin, y, SIGNATORY, data.issued_at[:10], data.cert_id)

    # Footer.
    c.setFont("Helvetica", 7.8)
    c.setFillColorRGB(*MUTED)
    note = (
        "Verify: rc-certificate --verify --report compliance_report.json. This is a project "
        "verification record of the checks listed, not a legal certification of GDPR compliance."
    )
    fy = 44
    for line in reversed(_wrap(note, "Helvetica", 7.8, width - 2 * margin, c)):
        c.drawString(margin, fy, line)
        fy += 10
    c.showPage()
    c.save()


def sidecar(data: CertificateData, pdf_path: Path) -> dict:
    return {
        "project": PROJECT,
        "certificate_id": data.cert_id,
        "report": "compliance_report.json",
        "overall_verdict": data.verdict,
        "sha256_report": data.digest,
        "algorithm": ALGORITHM if data.signature else None,
        "hmac_signature": data.signature,
        "signed_value": "the sha256_report hex string",
        "sha256_pdf": sha256_file(pdf_path),
        "issued_at": data.issued_at,
    }


@dataclass(frozen=True, slots=True)
class VerifyResult:
    report_matches: bool
    signature_valid: bool | None
    pdf_matches: bool | None
    pdf_embeds_hash: bool | None

    @property
    def passed(self) -> bool:
        return (
            self.report_matches
            and self.signature_valid is not False
            and self.pdf_matches is not False
            and self.pdf_embeds_hash is not False
        )

    def as_dict(self) -> dict:
        return {
            "report_matches": self.report_matches,
            "signature_valid": self.signature_valid,
            "pdf_matches": self.pdf_matches,
            "pdf_embeds_hash": self.pdf_embeds_hash,
            "verdict": "PASS" if self.passed else "FAIL",
        }


def verify_certificate(
    report_path: Path, sidecar_path: Path, pdf_path: Path | None, key: bytes | None
) -> VerifyResult:
    record = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    digest = sha256_file(report_path)
    report_matches = hmac.compare_digest(digest, str(record.get("sha256_report", "")))
    signature_valid = None
    if key is not None and record.get("hmac_signature"):
        signature_valid = hmac.compare_digest(sign(digest, key), str(record["hmac_signature"]))
    pdf_matches = pdf_embeds = None
    if pdf_path is not None and Path(pdf_path).is_file():
        pdf_matches = hmac.compare_digest(sha256_file(pdf_path), str(record.get("sha256_pdf", "")))
        pdf_embeds = f"sha256:{digest}".encode("ascii") in Path(pdf_path).read_bytes()
    return VerifyResult(report_matches, signature_valid, pdf_matches, pdf_embeds)
