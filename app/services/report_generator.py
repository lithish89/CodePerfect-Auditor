"""
CodePerfect Auditor — Report Generator  (perfectly mapped to your codebase)
============================================================================

Called from main.py as:
    from app.services.report_generator import generate_audit_report
    pdf_bytes = generate_audit_report(record)

Where `record` comes from:
    record = get_audit_by_id(audit_id)      # audit_db.py
    patient = get_patient_by_audit(audit_id) # claims_db.py
    if patient:
        record["patient"] = patient

Exact field sources
───────────────────
audit_logs table  →  id, created_at, clinical_note, risk_level,
                     revenue_risk, action_required, matched_count,
                     missing_count, extra_count, total_ai_icd,
                     total_ai_cpt, total_human, warnings,
                     recommendations, status, entities

audit_codes rows  →  record["codes"] list
                     each: { code, code_type, category,
                              description, confidence, risk, note }
                     categories: ai_confirmed | matched | missing
                                  extra | manual_review
                     code_type : "ICD-10" | "CPT"
                     confidence: 0.0 to 1.0  (multiplied x100 for display %)

patient dict      →  record["patient"]
                     { name, phone, email, dob, gender, address,
                       insurance_code, insurance_org,
                       insurer_email, hospital_name,
                       policy_number (optional), adjuster (optional) }
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette ────────────────────────────────────────────────────────────────
DARK_GREEN   = colors.HexColor("#1a472a")
MID_GREEN    = colors.HexColor("#2d6a4f")
LIGHT_GREEN  = colors.HexColor("#d8f3dc")
ACCENT_GOLD  = colors.HexColor("#f0a500")
DANGER_RED   = colors.HexColor("#c0392b")
WARN_ORANGE  = colors.HexColor("#e67e22")
INFO_BLUE    = colors.HexColor("#2471a3")
PURPLE       = colors.HexColor("#7d3c98")
SECTION_BG   = colors.HexColor("#f0f4f0")
ROW_ALT      = colors.HexColor("#f7fbf7")
ROW_ALT_BLUE = colors.HexColor("#f0f6ff")
WHITE        = colors.white
GREY_TEXT    = colors.HexColor("#555555")
DARK_TEXT    = colors.HexColor("#111111")
BORDER_GREY  = colors.HexColor("#cccccc")
LIGHT_GREY   = colors.HexColor("#e8e8e8")

PAGE_W, PAGE_H = A4
MARGIN    = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
HEADER_H  = 48 * mm
FOOTER_H  = 12 * mm


# ══════════════════════════════════════════════════════════════════════════════
#  Numbered canvas  (Page X of Y)
# ══════════════════════════════════════════════════════════════════════════════

class _NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._stamp_page(total)
            super().showPage()
        super().save()

    def _stamp_page(self, total: int):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(GREY_TEXT)
        self.drawRightString(
            PAGE_W - MARGIN, 6.5 * mm,
            f"Page {self._pageNumber} of {total}"
        )
        self.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
#  Canvas page decorations
# ══════════════════════════════════════════════════════════════════════════════

def _paint_header(canv, doc, record: dict):
    """Dark-green header bar painted on every page."""
    canv.saveState()

    canv.setFillColor(DARK_GREEN)
    canv.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, stroke=0, fill=1)

    canv.setFillColor(ACCENT_GOLD)
    canv.rect(0, PAGE_H - HEADER_H, PAGE_W, 3, stroke=0, fill=1)

    lx = MARGIN
    ty = PAGE_H - 13 * mm

    canv.setFillColor(WHITE)
    canv.setFont("Helvetica-Bold", 21)
    canv.drawString(lx, ty, "CodePerfect Auditor")

    canv.setFont("Helvetica", 9.5)
    canv.setFillColor(LIGHT_GREEN)
    canv.drawString(lx, ty - 13, "Medical Coding Compliance Audit Report")

    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.HexColor("#8ecfa0"))
    ts = record.get("created_at", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_fmt = dt.strftime("%d/%m/%Y  %I:%M %p UTC")
    except Exception:
        ts_fmt = datetime.now().strftime("%d/%m/%Y  %I:%M %p")
    canv.drawString(lx, ty - 25, f"Generated: {ts_fmt}")

    vx = PAGE_W - 72 * mm
    canv.setStrokeColor(colors.HexColor("#3a7a4f"))
    canv.setLineWidth(0.8)
    canv.line(vx, PAGE_H - 10 * mm, vx, PAGE_H - HEADER_H + 10 * mm)

    rx  = PAGE_W - MARGIN
    pat = record.get("patient") or {}
    hosp = pat.get("hospital_name", "")

    canv.setFont("Helvetica-Bold", 7.5)
    canv.setFillColor(ACCENT_GOLD)
    canv.drawRightString(rx, ty, "FACILITY")

    canv.setFont("Helvetica", 8.5)
    canv.setFillColor(WHITE)
    canv.drawRightString(rx, ty - 12, hosp or "—")

    audit_id = record.get("id", "")
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(colors.HexColor("#8ecfa0"))
    short_id = (audit_id[:16] + "...") if len(audit_id) > 16 else audit_id
    canv.drawRightString(rx, ty - 24, f"Audit ID: {short_id}")

    status = str(record.get("status", "")).upper()
    canv.setFont("Helvetica-Bold", 7.5)
    canv.setFillColor(ACCENT_GOLD)
    canv.drawRightString(rx, ty - 36, f"STATUS: {status}")

    canv.setFont("Helvetica-Oblique", 7)
    canv.setFillColor(colors.HexColor("#6db88a"))
    canv.drawString(lx, PAGE_H - HEADER_H + 6 * mm,
                    "CONFIDENTIAL — For authorised medical billing personnel only")

    canv.restoreState()


def _paint_footer(canv, doc):
    canv.saveState()
    canv.setStrokeColor(DARK_GREEN)
    canv.setLineWidth(1)
    canv.line(MARGIN, FOOTER_H, PAGE_W - MARGIN, FOOTER_H)
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(GREY_TEXT)
    canv.drawString(MARGIN, FOOTER_H - 8,
                    "CodePerfect Auditor — Confidential Medical Coding Audit Report")
    canv.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Flowables
# ══════════════════════════════════════════════════════════════════════════════

class SectionHeader(Flowable):
    def __init__(self, title: str, subtitle: str = "", width: float = CONTENT_W):
        super().__init__()
        self.title    = title
        self.subtitle = subtitle
        self.width    = width
        self.height   = 28 if subtitle else 20

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(SECTION_BG)
        c.roundRect(0, 0, self.width, self.height, 3, stroke=0, fill=1)
        c.setFillColor(DARK_GREEN)
        c.rect(0, 0, 4, self.height, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(DARK_GREEN)
        c.drawString(12, self.height - 13, self.title)
        if self.subtitle:
            c.setFont("Helvetica", 7.5)
            c.setFillColor(GREY_TEXT)
            c.drawString(12, 5, self.subtitle)
        c.restoreState()

    def wrap(self, aW, aH):
        return self.width, self.height


class RiskBadge(Flowable):
    _MAP = {
        "HIGH":   (colors.HexColor("#fdecea"), DANGER_RED),
        "MEDIUM": (colors.HexColor("#fff3e0"), WARN_ORANGE),
        "LOW":    (colors.HexColor("#e8f5e9"), MID_GREEN),
        "CLEAR":  (colors.HexColor("#e8f5e9"), MID_GREEN),
    }

    def __init__(self, level: str):
        super().__init__()
        self.level = level.upper()
        self.bg, self.fg = self._MAP.get(self.level, (LIGHT_GREY, GREY_TEXT))
        self.width  = 80
        self.height = 20

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self.height, 10, stroke=0, fill=1)
        c.setStrokeColor(self.fg)
        c.setLineWidth(0.8)
        c.roundRect(0, 0, self.width, self.height, 10, stroke=1, fill=0)
        c.setFillColor(self.fg)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(self.width / 2, 6, f"  {self.level} RISK")
        c.restoreState()

    def wrap(self, aW, aH):
        return self.width, self.height


class ConfidenceBar(Flowable):
    """Accepts confidence as 0.0-1.0 (from DB) or 0-100."""

    def __init__(self, value: float, bar_width: int = 58):
        super().__init__()
        pct = value if value > 1.0 else value * 100
        self.pct       = max(0, min(100, round(pct)))
        self.bar_width = bar_width
        self.height    = 12

    def draw(self):
        c = self.canv
        c.saveState()
        track_w = self.bar_width - 30
        c.setFillColor(LIGHT_GREY)
        c.roundRect(0, 3, track_w, 5, 2, stroke=0, fill=1)
        fill = max(1, int(track_w * self.pct / 100))
        col  = MID_GREEN if self.pct >= 85 else (WARN_ORANGE if self.pct >= 60 else DANGER_RED)
        c.setFillColor(col)
        c.roundRect(0, 3, fill, 5, 2, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(GREY_TEXT)
        c.drawString(track_w + 4, 2, f"{self.pct}%")
        c.restoreState()

    def wrap(self, aW, aH):
        return self.bar_width, self.height


class CodeBlock(Flowable):
    def __init__(self, code: str, description: str, tag: str,
                 risk_label: str, accent: colors.Color,
                 note: str = "", width: float = CONTENT_W):
        super().__init__()
        self.code        = code
        self.description = description
        self.tag         = tag
        self.risk_label  = risk_label
        self.accent      = accent
        self.note        = note
        self.width       = width
        self.height      = 36 if note else 28

    def draw(self):
        c = self.canv
        c.saveState()

        bg = colors.Color(self.accent.red, self.accent.green,
                          self.accent.blue, alpha=0.07)
        c.setFillColor(bg)
        c.roundRect(0, 0, self.width, self.height, 3, stroke=0, fill=1)

        c.setFillColor(self.accent)
        c.rect(0, 0, 4, self.height, stroke=0, fill=1)

        pill_h = 13
        pill_w = len(self.tag) * 5.5 + 10
        pill_y = self.height - 16
        c.setFillColor(self.accent)
        c.roundRect(10, pill_y, pill_w, pill_h, 3, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(WHITE)
        c.drawCentredString(10 + pill_w / 2, pill_y + 3, self.tag)

        c.setFont("Helvetica-Bold", 10.5)
        c.setFillColor(self.accent)
        c.drawString(16 + pill_w, self.height - 14, self.code)

        c.setFont("Helvetica", 9)
        c.setFillColor(DARK_TEXT)
        desc = (self.description[:72] + "...") if len(self.description) > 72 else self.description
        c.drawString(16 + pill_w + 58, self.height - 14, desc)

        rpl_w = len(self.risk_label) * 5 + 14
        rpx   = self.width - rpl_w - 6
        c.setFillColor(colors.Color(self.accent.red, self.accent.green,
                                    self.accent.blue, alpha=0.15))
        c.roundRect(rpx, self.height - 17, rpl_w, 13, 6, stroke=0, fill=1)
        c.setFont("Helvetica", 7)
        c.setFillColor(self.accent)
        c.drawCentredString(rpx + rpl_w / 2, self.height - 14, self.risk_label)

        if self.note:
            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColor(GREY_TEXT)
            note_t = (self.note[:95] + "...") if len(self.note) > 95 else self.note
            c.drawString(14, 6, note_t)

        c.restoreState()

    def wrap(self, aW, aH):
        return self.width, self.height


# ══════════════════════════════════════════════════════════════════════════════
#  Style helpers
# ══════════════════════════════════════════════════════════════════════════════

def _S() -> dict:
    return {
        "lbl":   ParagraphStyle("lbl",   fontSize=8,   fontName="Helvetica",
                                textColor=GREY_TEXT),
        "val":   ParagraphStyle("val",   fontSize=9,   fontName="Helvetica-Bold",
                                textColor=DARK_TEXT),
        "norm":  ParagraphStyle("norm",  fontSize=9,   fontName="Helvetica",
                                textColor=colors.HexColor("#333333"), leading=13),
        "sml":   ParagraphStyle("sml",   fontSize=7.5, fontName="Helvetica",
                                textColor=GREY_TEXT),
        "thhd":  ParagraphStyle("thhd",  fontSize=8.5, fontName="Helvetica-Bold",
                                textColor=WHITE),
        "tcell": ParagraphStyle("tcell", fontSize=8.5, fontName="Helvetica",
                                textColor=DARK_TEXT),
        "tcode": ParagraphStyle("tcode", fontSize=9,   fontName="Helvetica-Bold",
                                textColor=DARK_GREEN),
        "tcpt":  ParagraphStyle("tcpt",  fontSize=9,   fontName="Helvetica-Bold",
                                textColor=INFO_BLUE),
        "tpur":  ParagraphStyle("tpur",  fontSize=9,   fontName="Helvetica-Bold",
                                textColor=PURPLE),
        "action":ParagraphStyle("action",fontSize=9,   fontName="Helvetica-Bold",
                                textColor=DANGER_RED),
    }


def _kv(label: str, value, S: dict):
    return [Paragraph(label, S["lbl"]),
            Paragraph(str(value) if value else "—", S["val"])]


def _two_col(left_rows: list, right_rows: list) -> Table:
    cw = CONTENT_W / 2
    col_w = [cw * 0.36, cw * 0.64, cw * 0.36, cw * 0.64]
    rows = []
    n = max(len(left_rows), len(right_rows))
    for i in range(n):
        l = left_rows[i]  if i < len(left_rows)  else ["", ""]
        r = right_rows[i] if i < len(right_rows) else ["", ""]
        rows.append(l + r)
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, BORDER_GREY),
        ("LINEAFTER",     (1, 0), (1, -1),  0.5, BORDER_GREY),
        ("LEFTPADDING",   (2, 0), (2, -1),  12),
    ]))
    return t


def _stat_card(label: str, value, colour: colors.Color) -> Table:
    t = Table(
        [[Paragraph(str(value), ParagraphStyle(
             "sv", fontSize=15, fontName="Helvetica-Bold",
             textColor=colour, alignment=TA_CENTER))],
         [Paragraph(label, ParagraphStyle(
             "sl", fontSize=7, fontName="Helvetica",
             textColor=GREY_TEXT, alignment=TA_CENTER))]],
        colWidths=[38 * mm],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.6, BORDER_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _code_table(rows: list, header_colour: colors.Color,
                row_alt: colors.Color, col_widths: list,
                headers: list, S: dict) -> Table:
    hdr = [Paragraph(h, S["thhd"]) for h in headers]
    body = [hdr] + rows
    t = Table(body, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), header_colour),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, row_alt]),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_GREY),
        ("LINEBELOW",     (0, 0), (-1,  0), 1.2, ACCENT_GOLD),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _parse_revenue(rev_str: str):
    """Split 'range' string from auditor_agent into (lost, penalty)."""
    if not rev_str:
        return "—", "—"
    parts = rev_str.split("·")
    lost    = parts[0].strip().split(" ")[0] if parts else "—"
    penalty = parts[1].strip().split(" ")[0] if len(parts) > 1 else "—"
    return lost, penalty


# ══════════════════════════════════════════════════════════════════════════════
#  Main generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_audit_report(record: dict) -> bytes:
    """
    Build and return PDF bytes for a single audit record.

    Drop-in replacement for the original generate_audit_report().
    No signature change — still accepts the same `record` dict that
    main.py already constructs and passes.
    """
    buffer = io.BytesIO()
    S      = _S()

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=HEADER_H + 6 * mm,
        bottomMargin=FOOTER_H + 8 * mm,
    )
    frame = Frame(
        MARGIN, FOOTER_H + 8 * mm,
        CONTENT_W, PAGE_H - HEADER_H - FOOTER_H - 14 * mm,
        id="main",
    )

    def _on_page(canv, doc):
        _paint_header(canv, doc, record)
        _paint_footer(canv, doc)

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])
    story = []

    def sp(h=6):
        story.append(Spacer(1, h))

    def sec(title, subtitle=""):
        story.append(SectionHeader(title, subtitle))
        sp(5)

    # ── helper: filter codes list by category + optional code_type ──────
    codes: list[dict] = record.get("codes", [])

    def _codes(category: str, code_type: str = "") -> list[dict]:
        return [c for c in codes
                if c.get("category") == category
                and (not code_type or c.get("code_type") == code_type)]

    # ══════════════════════════════════════════════════════════════════════
    # 1.  PATIENT  +  INSURER  DETAILS
    #     Fields: name, dob, gender, phone, email, address, insurance_code
    #             insurance_org, insurer_email, policy_number, adjuster,
    #             hospital_name
    #     Source: record["patient"]  from  claims_db.get_patient_by_audit()
    # ══════════════════════════════════════════════════════════════════════
    pat = record.get("patient") or {}

    sec("PATIENT & INSURER DETAILS",
        "Demographic, contact, and insurance identifiers")

    left = [
        _kv("Full Name",       pat.get("name"),           S),
        _kv("Date of Birth",   pat.get("dob"),            S),
        _kv("Gender",          pat.get("gender"),         S),
        _kv("Phone",           pat.get("phone"),          S),
        _kv("Email",           pat.get("email"),          S),
        _kv("Address",         pat.get("address"),        S),
        _kv("Insurance Code",  pat.get("insurance_code"), S),
    ]
    policy_number = pat.get("policy_number") or pat.get("insurance_code") or ""
    adjuster = pat.get("adjuster") or ""

    right = [
        _kv("Insurance Org",   pat.get("insurance_org"),  S),
        _kv("Claims Email",    pat.get("insurer_email"),  S),
        _kv("Policy Number",   policy_number,             S),
        _kv("Adjuster",        adjuster,                  S),
        _kv("Facility",        pat.get("hospital_name"),  S),
    ]
    story.append(_two_col(left, right))
    sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 2.  AUDIT SUMMARY
    #     Fields from audit_logs table:
    #       risk_level, matched_count, missing_count, extra_count,
    #       total_ai_icd, total_ai_cpt, total_human,
    #       revenue_risk (range string), action_required
    # ══════════════════════════════════════════════════════════════════════
    sec("AUDIT SUMMARY", "Risk assessment and financial impact overview")

    risk          = str(record.get("risk_level", "low")).upper()
    matched_n     = record.get("matched_count",  0)
    missing_n     = record.get("missing_count",  0)
    extra_n       = record.get("extra_count",    0)
    ai_icd_n      = record.get("total_ai_icd",   0)
    ai_cpt_n      = record.get("total_ai_cpt",   0)
    human_n       = record.get("total_human",    0)
    action        = record.get("action_required", "—")
    rev_lost, rev_penalty = _parse_revenue(record.get("revenue_risk", ""))

    badge_tbl = Table(
        [[RiskBadge(risk)],
         [Spacer(1, 5)],
         [Paragraph("Action Required", S["lbl"])],
         [Paragraph(action, S["action"])]],
        colWidths=[50 * mm],
    )
    badge_tbl.setStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])

    stat_items = [
        ("Matched\nCodes",       matched_n,  MID_GREEN),
        ("Missing\nCodes",       missing_n,  DANGER_RED),
        ("Extra\nCodes",         extra_n,    WARN_ORANGE),
        ("AI ICD-10\nGenerated", ai_icd_n,   INFO_BLUE),
        ("AI CPT\nGenerated",    ai_cpt_n,   PURPLE),
    ]
    cards = [_stat_card(lbl, val, col) for lbl, val, col in stat_items]
    cards_tbl = Table([cards], colWidths=[38 * mm] * 5)
    cards_tbl.setStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ])

    outer = Table(
        [[badge_tbl, cards_tbl]],
        colWidths=[54 * mm, CONTENT_W - 54 * mm],
    )
    outer.setStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ])
    story.append(outer)
    sp(6)

    # Revenue / penalty / human codes bar
    fin_row = [[
        Paragraph("Revenue Lost Est.",  S["lbl"]),
        Paragraph(rev_lost, ParagraphStyle("rl", fontSize=9, fontName="Helvetica-Bold",
                                           textColor=DANGER_RED)),
        Paragraph("Penalty Risk Est.",  S["lbl"]),
        Paragraph(rev_penalty, ParagraphStyle("pe", fontSize=9, fontName="Helvetica-Bold",
                                              textColor=WARN_ORANGE)),
        Paragraph("Human Codes Submitted", S["lbl"]),
        Paragraph(str(human_n), ParagraphStyle("hc", fontSize=9, fontName="Helvetica-Bold",
                                               textColor=INFO_BLUE)),
    ]]
    cw6 = CONTENT_W / 6
    fin_tbl = Table(fin_row, colWidths=[cw6] * 6)
    fin_tbl.setStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), SECTION_BG),
        ("BOX",           (0, 0), (-1, -1), 0.6, BORDER_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("LINEAFTER",     (1, 0), (1, 0),  0.5, BORDER_GREY),
        ("LINEAFTER",     (3, 0), (3, 0),  0.5, BORDER_GREY),
    ])
    story.append(fin_tbl)
    sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 3.  CLINICAL NOTE
    #     Field: record["clinical_note"]  (truncated at 2000 chars in DB)
    # ══════════════════════════════════════════════════════════════════════
    sec("CLINICAL NOTE", "Source documentation used for AI code generation")

    note_text = record.get("clinical_note", "No clinical note available.")
    note_box = Table(
        [[Paragraph(note_text.replace("\n", "<br/>"), S["norm"])]],
        colWidths=[CONTENT_W],
    )
    note_box.setStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f9fbf9")),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#c3d9c8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ])
    story.append(note_box)
    sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 4.  AI-GENERATED ICD-10 CODES
    #     Source: codes  where  category="ai_confirmed"  AND  code_type="ICD-10"
    #     Saved by save_audit() → _insert_codes(ai_codes["icd_codes"], "ai_confirmed")
    # ══════════════════════════════════════════════════════════════════════
    ai_icd = _codes("ai_confirmed", "ICD-10")
    if ai_icd:
        sec(f"AI-GENERATED ICD-10 CODES  ({len(ai_icd)})",
            "Diagnosis codes extracted from clinical note  |  confidence >= 50%")
        cw_code = 22 * mm
        cw_conf = 60
        cw_desc = CONTENT_W - cw_code - cw_conf
        rows = []
        for c in ai_icd:
            rows.append([
                Paragraph(c.get("code", ""),        S["tcode"]),
                Paragraph(c.get("description", ""), S["tcell"]),
                ConfidenceBar(c.get("confidence") or 0),
            ])
        story.append(_code_table(
            rows, DARK_GREEN, ROW_ALT,
            [cw_code, cw_desc, cw_conf],
            ["Code", "Description", "Confidence"], S,
        ))
        sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 5.  AI-GENERATED CPT CODES
    #     Source: codes  where  category="ai_confirmed"  AND  code_type="CPT"
    #     Saved by save_audit() → _insert_codes(ai_codes["cpt_codes"], "ai_confirmed")
    # ══════════════════════════════════════════════════════════════════════
    ai_cpt = _codes("ai_confirmed", "CPT")
    if ai_cpt:
        sec(f"AI-GENERATED CPT CODES  ({len(ai_cpt)})",
            "Procedure codes extracted from clinical note")
        cw_code = 22 * mm
        cw_conf = 60
        cw_desc = CONTENT_W - cw_code - cw_conf
        rows = []
        for c in ai_cpt:
            rows.append([
                Paragraph(c.get("code", ""),        S["tcpt"]),
                Paragraph(c.get("description", ""), S["tcell"]),
                ConfidenceBar(c.get("confidence") or 0),
            ])
        story.append(_code_table(
            rows, INFO_BLUE, ROW_ALT_BLUE,
            [cw_code, cw_desc, cw_conf],
            ["Code", "Description", "Confidence"], S,
        ))
        sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 6.  MATCHED CODES
    #     Source: codes  where  category="matched"
    #     From audit_result["matched"]  via  auditor_agent.audit_codes()
    # ══════════════════════════════════════════════════════════════════════
    matched_codes = _codes("matched")
    if matched_codes:
        sec(f"MATCHED CODES  ({len(matched_codes)})",
            "Codes confirmed by both AI engine and human coder")
        cw_code = 22 * mm
        cw_type = 18 * mm
        cw_conf = 58
        cw_desc = CONTENT_W - cw_code - cw_type - cw_conf
        rows = []
        for c in matched_codes:
            rows.append([
                Paragraph(c.get("code", ""),      S["tcode"]),
                Paragraph(c.get("code_type", ""), S["sml"]),
                Paragraph(c.get("description",""),S["tcell"]),
                ConfidenceBar(c.get("confidence") or 0),
            ])
        story.append(_code_table(
            rows, MID_GREEN, ROW_ALT,
            [cw_code, cw_type, cw_desc, cw_conf],
            ["Code", "Type", "Description", "Confidence"], S,
        ))
        sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 7.  MISSING CODES  —  UNDERCODING RISK
    #     Source: codes  where  category="missing"
    #     From audit_result["missing_codes"]  via  auditor_agent.audit_codes()
    #     Fields used: code, code_type, description, risk, note
    # ══════════════════════════════════════════════════════════════════════
    missing = _codes("missing")
    if missing:
        sec(f"MISSING CODES — UNDERCODING RISK  ({len(missing)})",
            "AI identified these codes in the note  |  absent from human submission")

        RISK_LBL = {"high": "HIGH risk", "medium": "MEDIUM risk", "low": "LOW risk"}

        for c in missing:
            ct        = c.get("code_type", "ICD-10")
            risk_lbl  = RISK_LBL.get(str(c.get("risk", "low")).lower(), "LOW risk")
            story.append(KeepTogether([
                CodeBlock(
                    code        = c.get("code", ""),
                    description = c.get("description", ""),
                    tag         = ct,
                    risk_label  = f"{ct}  {risk_lbl}",
                    accent      = DANGER_RED,
                    note        = c.get("note", ""),
                ),
                Spacer(1, 4),
            ]))
        sp(6)

    # ══════════════════════════════════════════════════════════════════════
    # 8.  EXTRA CODES  —  UPCODING RISK
    #     Source: codes  where  category="extra"
    #     From audit_result["extra_codes"]  via  auditor_agent.audit_codes()
    # ══════════════════════════════════════════════════════════════════════
    extra = _codes("extra")
    if extra:
        sec(f"EXTRA CODES — UPCODING RISK  ({len(extra)})",
            "Human-submitted codes  |  not supported by AI or clinical note")

        for c in extra:
            ct       = c.get("code_type", "ICD-10")
            risk_lbl = RISK_LBL.get(str(c.get("risk", "low")).lower(), "LOW risk")
            story.append(KeepTogether([
                CodeBlock(
                    code        = c.get("code", ""),
                    description = c.get("description", ""),
                    tag         = ct,
                    risk_label  = f"{ct}  {risk_lbl}",
                    accent      = WARN_ORANGE,
                    note        = c.get("note", ""),
                ),
                Spacer(1, 4),
            ]))
        sp(6)

    # ══════════════════════════════════════════════════════════════════════
    # 9.  MANUAL REVIEW  (low-confidence AI codes)
    #     Source: codes  where  category="manual_review"
    #     From ai_codes["manual_review"]  via  coding_logic.generate_icd_codes()
    # ══════════════════════════════════════════════════════════════════════
    manual = _codes("manual_review")
    if manual:
        sec(f"MANUAL REVIEW  ({len(manual)})",
            "Low-confidence AI suggestions — verify against clinical note before use")

        cw_code = 22 * mm
        cw_conf = 58
        cw_note = 42 * mm
        cw_desc = CONTENT_W - cw_code - cw_conf - cw_note
        rows = []
        for c in manual:
            rows.append([
                Paragraph(c.get("code", ""),        S["tpur"]),
                Paragraph(c.get("description", ""), S["tcell"]),
                ConfidenceBar(c.get("confidence") or 0),
                Paragraph(c.get("note", "—"),        S["sml"]),
            ])
        story.append(_code_table(
            rows, PURPLE, colors.HexColor("#f5eef8"),
            [cw_code, cw_desc, cw_conf, cw_note],
            ["Code", "Description", "Confidence", "Note"], S,
        ))
        sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 10.  RECOMMENDATIONS
    #      Source: record["recommendations"]
    #      Stored as JSON list string in DB; already parsed by get_audit_by_id()
    #      Generated by auditor_agent.audit_codes() → recommendations list
    # ══════════════════════════════════════════════════════════════════════
    recs = record.get("recommendations", [])
    if isinstance(recs, str):
        import json as _json
        try:    recs = _json.loads(recs)
        except: recs = [recs]

    if recs:
        sec("RECOMMENDATIONS", "AI-generated action items for the billing team")
        rec_items = []
        for r in recs:
            rec_items.append(Paragraph(
                f"&#x2192;  {r}",
                ParagraphStyle("ri", fontSize=8.5, fontName="Helvetica",
                               textColor=colors.HexColor("#1a3a1a"),
                               leading=13, spaceAfter=3, leftIndent=6),
            ))
        rec_box = Table([[item] for item in rec_items], colWidths=[CONTENT_W])
        rec_box.setStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#eaf6ee")),
            ("BOX",           (0, 0), (-1, -1), 0.6, MID_GREEN),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ])
        story.append(rec_box)
        sp(10)

    # ══════════════════════════════════════════════════════════════════════
    # 11.  WARNINGS
    #      Source: record["warnings"]
    #      Stored as JSON list; parsed by get_audit_by_id()
    #      Generated by coding_logic.generate_icd_codes() → warnings list
    # ══════════════════════════════════════════════════════════════════════
    warnings = record.get("warnings", [])
    if isinstance(warnings, str):
        import json as _json
        try:    warnings = _json.loads(warnings)
        except: warnings = [warnings]

    if warnings:
        sec(f"WARNINGS  ({len(warnings)})",
            "CMS rule flags and AI engine notices — review before submission")
        warn_items = []
        for w in warnings:
            warn_items.append(Paragraph(
                f"  {w}",
                ParagraphStyle("wi", fontSize=8.5, fontName="Helvetica",
                               textColor=colors.HexColor("#7d3c00"),
                               leading=13, spaceAfter=3, leftIndent=6),
            ))
        warn_box = Table([[item] for item in warn_items], colWidths=[CONTENT_W])
        warn_box.setStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fff8e1")),
            ("BOX",           (0, 0), (-1, -1), 0.6, WARN_ORANGE),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ])
        story.append(warn_box)
        sp(8)

    # ── Build ──────────────────────────────────────────────────────────────
    doc.build(story, canvasmaker=_NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke-test  —  python report_generator.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SAMPLE = {
        "id":             "f85cf6b3-1234-4abc-8def-000000000001",
        "created_at":     "2026-03-24T10:06:20Z",
        "clinical_note": (
            "Patient is a 67-year-old male with SOB and chest pain.\n"
            "History of HTN and DM2. No history of cancer.\n"
            "CT chest and CBC ordered. IV infusion of antibiotics started for CAP.\n"
            "No evidence of PE on CT."
        ),
        "risk_level":      "high",
        "revenue_risk":    "~$1,700 potential revenue loss · $0 penalty risk",
        "action_required": "URGENT — bill amendment required",
        "matched_count":   3,
        "missing_count":   5,
        "extra_count":     1,
        "total_ai_icd":    5,
        "total_ai_cpt":    3,
        "total_human":     4,
        "status":          "reviewed",
        "warnings":  ["No ICD-10 match for 'SOB' — needs manual review."],
        "recommendations": [
            "5 potentially missing code(s) detected. Estimated revenue loss: ~$1,700. "
            "Review and add to avoid undercoding.",
            "1 potentially unsupported code(s) flagged. Estimated penalty exposure: $0. "
            "Verify clinical documentation before submitting.",
        ],
        "entities": {
            "diagnoses":  ["chest pain", "hypertension", "type 2 diabetes", "pneumonia"],
            "procedures": ["CT scan of chest", "blood test", "IV infusion"],
            "symptoms":   ["shortness of breath"],
        },
        # ─── audit_codes table rows ────────────────────────────────────────
        # confidence stored as 0.0–1.0 in your DB (real column type)
        "codes": [
            # AI-confirmed ICD-10  (category="ai_confirmed", code_type="ICD-10")
            {"code":"R07.9", "code_type":"ICD-10","category":"ai_confirmed",
             "description":"Chest pain, unspecified",                        "confidence":0.97,"risk":"low",   "note":""},
            {"code":"E11.9", "code_type":"ICD-10","category":"ai_confirmed",
             "description":"Type 2 diabetes mellitus without complications",  "confidence":0.97,"risk":"medium","note":""},
            {"code":"I10",   "code_type":"ICD-10","category":"ai_confirmed",
             "description":"Essential (primary) hypertension",               "confidence":0.97,"risk":"medium","note":""},
            {"code":"J18.9", "code_type":"ICD-10","category":"ai_confirmed",
             "description":"Pneumonia, unspecified organism",                 "confidence":0.97,"risk":"medium","note":""},
            {"code":"R06.09","code_type":"ICD-10","category":"ai_confirmed",
             "description":"Other forms of dyspnoea",                        "confidence":0.97,"risk":"low",   "note":""},
            # AI-confirmed CPT  (category="ai_confirmed", code_type="CPT")
            {"code":"85025", "code_type":"CPT",   "category":"ai_confirmed",
             "description":"Blood test — CBC with differential",              "confidence":0.90,"risk":"low",   "note":""},
            {"code":"71250", "code_type":"CPT",   "category":"ai_confirmed",
             "description":"CT thorax — chest scan",                         "confidence":0.90,"risk":"low",   "note":""},
            {"code":"96365", "code_type":"CPT",   "category":"ai_confirmed",
             "description":"IV infusion, initial, antibiotic therapy",       "confidence":0.90,"risk":"low",   "note":""},
            # Matched  (category="matched")
            {"code":"J18.9", "code_type":"ICD-10","category":"matched",
             "description":"Pneumonia, unspecified organism",                 "confidence":0.97,"risk":"medium","note":""},
            {"code":"I10",   "code_type":"ICD-10","category":"matched",
             "description":"Essential (primary) hypertension",               "confidence":0.97,"risk":"medium","note":""},
            {"code":"85025", "code_type":"CPT",   "category":"matched",
             "description":"Blood test — CBC",                                "confidence":0.90,"risk":"low",   "note":""},
            # Missing  (category="missing")
            {"code":"R07.9", "code_type":"ICD-10","category":"missing",
             "description":"Chest pain, unspecified",
             "confidence":0.92,"risk":"low",
             "note":"Identified by AI but not entered by human coder."},
            {"code":"E11.9", "code_type":"ICD-10","category":"missing",
             "description":"Type 2 diabetes mellitus without complications",
             "confidence":0.89,"risk":"high",
             "note":"High-impact undercoding — DM2 substantially affects DRG weight."},
            {"code":"R06.09","code_type":"ICD-10","category":"missing",
             "description":"Other forms of dyspnoea",                        "confidence":0.91,"risk":"low",   "note":""},
            {"code":"71250", "code_type":"CPT",   "category":"missing",
             "description":"CT scan — chest",                                 "confidence":0.93,"risk":"medium","note":""},
            {"code":"96365", "code_type":"CPT",   "category":"missing",
             "description":"IV infusion — initial",                          "confidence":0.91,"risk":"low",   "note":""},
            # Extra  (category="extra")
            {"code":"Z87.01","code_type":"ICD-10","category":"extra",
             "description":"History of pneumonia",
             "confidence":0.0,"risk":"low",
             "note":"Entered by human coder but not supported by current clinical note."},
            # Manual review  (category="manual_review")
            {"code":"R06.00","code_type":"ICD-10","category":"manual_review",
             "description":"Dyspnoea, unspecified",
             "confidence":0.42,"risk":"low",
             "note":"Low confidence (0.42) — verify manually before use"},
        ],
        "patient": {
            "name":           "Rajan Kumar",
            "dob":            "15/03/1957",
            "gender":         "Male",
            "phone":          "+91 98765 43210",
            "email":          "rajan.kumar@email.com",
            "address":        "45 Anna Nagar, Chennai - 600040",
            "insurance_code": "INS-2024-00456",
            "insurance_org":  "Star Health Insurance",
            "insurer_email":  "claims@starhealth.in",
            "policy_number":  "POL-2024-78923",
            "adjuster":       "Meena Iyer",
            "hospital_name":  "Apollo Hospitals, Chennai",
        },
    }

    pdf = generate_audit_report(SAMPLE)
    out = "/mnt/user-data/outputs/audit_report.pdf"
    with open(out, "wb") as f:
        f.write(pdf)
    print(f"Wrote {len(pdf):,} bytes to {out}")