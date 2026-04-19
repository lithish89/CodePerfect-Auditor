"""
Email Service  —  app/services/email_service.py
-------------------------------------------------
Sends audit reports and claim notifications via SMTP.

Setup (Gmail example):
  1. Enable 2FA on your Gmail account
  2. Create an App Password: Google Account → Security → App Passwords
  3. Set environment variables:
       SMTP_HOST=smtp.gmail.com
       SMTP_PORT=587
       SMTP_USER=your@gmail.com
       SMTP_PASS=your-app-password
       SMTP_FROM=your@gmail.com

For Outlook / Office 365:
       SMTP_HOST=smtp.office365.com
       SMTP_PORT=587

For development (no real email):
  Leave SMTP_USER empty — emails are logged to console instead of sent.
"""

from __future__ import annotations

import os
import smtplib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from typing import Optional

# ── Config (from environment variables) ───────────────────────────────────────
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "lithishkn248248@gmail.com")       # leave blank for dev mode
SMTP_PASS = os.environ.get("SMTP_PASS", "fnli llrd jvfr dpxe") 
SMTP_FROM = os.environ.get("SMTP_FROM", "lithishkn248248@gmail.com")

DEV_MODE  = not SMTP_USER   # True = log emails instead of sending


# ── Core send function ────────────────────────────────────────────────────────

def _send(
    to_emails: list[str],
    subject: str,
    html_body: str,
    attachments: list[tuple[bytes, str]] = None,  # [(bytes, filename)]
) -> tuple[bool, str]:
    """
    Send an email. Returns (success, message).
    In dev mode (no SMTP credentials), logs to console and returns success.
    """
    if DEV_MODE:
        print(f"\n[EmailService] DEV MODE — email not sent")
        print(f"  To:      {', '.join(to_emails)}")
        print(f"  Subject: {subject}")
        print(f"  Body:    {html_body[:200]}...")
        if attachments:
            print(f"  Attachments: {[a[1] for a in attachments]}")
        return True, "DEV MODE: email logged (not sent). Set SMTP_USER env var to send real emails."

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM
        msg["To"]      = ", ".join(to_emails)

        # HTML body
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # Attachments
        if attachments:
            for data, filename in attachments:
                part = MIMEApplication(data, Name=filename)
                part["Content-Disposition"] = f'attachment; filename="{filename}"'
                msg.attach(part)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, to_emails, msg.as_string())

        return True, f"Email sent to {', '.join(to_emails)}"

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check SMTP_USER and SMTP_PASS."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {e}"
    except Exception as e:
        return False, f"Email failed: {e}"


# ── Email templates ───────────────────────────────────────────────────────────

def _base_html(title: str, body: str) -> str:
    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#f4f3ef;padding:20px;margin:0">
    <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
      <div style="background:#1a472a;padding:24px 28px">
        <div style="font-size:22px;font-weight:800;color:#fff;letter-spacing:-0.5px">CodePerfect</div>
        <div style="font-size:12px;color:#9fe1cb;margin-top:2px">Medical Coding Auditor</div>
      </div>
      <div style="padding:28px">
        <h2 style="color:#1a1916;font-size:18px;margin:0 0 16px">{title}</h2>
        {body}
      </div>
      <div style="background:#f9f8f5;padding:14px 28px;font-size:11px;color:#8a8780;border-top:1px solid #e4e2db">
        This is an automated report from CodePerfect Auditor. Generated {datetime.utcnow().strftime('%d %b %Y, %H:%M UTC')}.
      </div>
    </div>
    </body></html>
    """


def send_patient_report(
    patient_email: str,
    patient_name: str,
    claim_id: str,
    audit_result: dict,
    pdf_bytes: Optional[bytes] = None,
) -> tuple[bool, str]:
    """Send audit report to patient."""
    ar  = audit_result or {}
    risk = ar.get("risk_level","unknown").upper()
    risk_color = "#b91c1c" if risk=="HIGH" else "#92400e" if risk=="MEDIUM" else "#14532d"

    body = f"""
    <p style="color:#4a4840;font-size:14px">Dear <strong>{patient_name}</strong>,</p>
    <p style="color:#4a4840;font-size:14px">Your medical claim audit has been completed. Please find the summary below:</p>

    <div style="background:#f9f8f5;border-radius:8px;padding:16px;margin:16px 0;border:1px solid #e4e2db">
      <table style="width:100%;font-size:13px;border-collapse:collapse">
        <tr><td style="color:#8a8780;padding:4px 0;width:40%">Claim ID</td>
            <td style="font-family:monospace;color:#1a1916">{claim_id[:8]}…</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Risk Level</td>
            <td style="font-weight:700;color:{risk_color}">{risk}</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Matched Codes</td>
            <td style="color:#14532d;font-weight:600">{ar.get('summary',{}).get('matched',0)}</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Revenue Impact</td>
            <td style="color:#1a1916">{ar.get('revenue_risk','—')}</td></tr>
      </table>
    </div>

    <p style="color:#4a4840;font-size:13px">
      {'The full audit report is attached as a PDF.' if pdf_bytes else 'Please contact your hospital for the full report.'}
    </p>
    <p style="color:#8a8780;font-size:12px;margin-top:20px">
      If you have questions about this claim, please contact your hospital billing department.
    </p>
    """

    attachments = []
    if pdf_bytes:
        attachments.append((pdf_bytes, f"audit_report_{claim_id[:8]}.pdf"))

    return _send(
        to_emails=[patient_email],
        subject=f"Your Medical Claim Report — CodePerfect Auditor",
        html_body=_base_html("Medical Claim Audit Report", body),
        attachments=attachments,
    )


def send_insurer_claim(
    insurer_email: str,
    insurer_org: str,
    patient_name: str,
    claim_id: str,
    audit_result: dict,
    ai_codes: dict,
    hospital_name: str,
    insurance_code: str,
    pdf_bytes: Optional[bytes] = None,
) -> tuple[bool, str]:
    """Send claim notification to insurer."""
    ar       = audit_result or {}
    icd_list = [c.get("icd_code","") for c in ai_codes.get("icd_codes",[])]
    cpt_list = [c.get("cpt_code","") for c in ai_codes.get("cpt_codes",[])]
    risk     = ar.get("risk_level","unknown").upper()
    risk_color = "#b91c1c" if risk=="HIGH" else "#92400e" if risk=="MEDIUM" else "#14532d"

    codes_html = ""
    if icd_list:
        codes_html += f"<p style='font-size:12px;color:#8a8780;margin:4px 0'>ICD-10: <strong style='color:#1a1916'>{', '.join(icd_list)}</strong></p>"
    if cpt_list:
        codes_html += f"<p style='font-size:12px;color:#8a8780;margin:4px 0'>CPT: <strong style='color:#1a1916'>{', '.join(cpt_list)}</strong></p>"

    body = f"""
    <p style="color:#4a4840;font-size:14px">Dear <strong>{insurer_org}</strong> Claims Team,</p>
    <p style="color:#4a4840;font-size:14px">A new insurance claim has been submitted for your review.</p>

    <div style="background:#f9f8f5;border-radius:8px;padding:16px;margin:16px 0;border:1px solid #e4e2db">
      <table style="width:100%;font-size:13px;border-collapse:collapse">
        <tr><td style="color:#8a8780;padding:4px 0;width:40%">Claim ID</td>
            <td style="font-family:monospace;color:#1a1916">{claim_id[:8]}…</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Patient</td>
            <td style="color:#1a1916">{patient_name}</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Insurance Code</td>
            <td style="color:#1a1916">{insurance_code}</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Hospital</td>
            <td style="color:#1a1916">{hospital_name}</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Risk Level</td>
            <td style="font-weight:700;color:{risk_color}">{risk}</td></tr>
        <tr><td style="color:#8a8780;padding:4px 0">Revenue at Risk</td>
            <td style="color:#1a1916">{ar.get('total_at_risk','—')}</td></tr>
      </table>
    </div>

    <div style="background:#f9f8f5;border-radius:8px;padding:14px 16px;margin:12px 0;border:1px solid #e4e2db">
      <p style="font-size:12px;font-weight:700;color:#1a1916;margin:0 0 6px">Coding Summary</p>
      {codes_html}
    </div>

    <p style="color:#4a4840;font-size:13px">
      The full audit report with detailed code justification is attached.
      Please log in to the CodePerfect portal to update the claim status.
    </p>
    """

    attachments = []
    if pdf_bytes:
        attachments.append((pdf_bytes, f"claim_{claim_id[:8]}.pdf"))

    return _send(
        to_emails=[insurer_email],
        subject=f"New Insurance Claim — {patient_name} [{insurance_code}]",
        html_body=_base_html("Insurance Claim Submission", body),
        attachments=attachments,
    )


def send_status_update(
    patient_email: str,
    patient_name: str,
    claim_id: str,
    new_status: str,
    remarks: str = "",
    insurer_org: str = "",
) -> tuple[bool, str]:
    """Notify patient when claim status changes."""
    status_map = {
        "accepted":             ("Your Claim Has Been Accepted",   "#14532d", "accepted and approved for reimbursement"),
        "rejected":             ("Your Claim Has Been Rejected",    "#b91c1c", "rejected"),
        "issued":               ("Your Claim Payment Issued",       "#14532d", "approved and payment has been issued"),
        "requires_verification":("Verification Required",           "#92400e", "under review and requires additional verification"),
        "pending":              ("Claim Status Update",             "#1e3a5f", "pending review"),
    }
    title, color, desc = status_map.get(new_status, ("Claim Status Update","#1a1916","updated"))

    body = f"""
    <p style="color:#4a4840;font-size:14px">Dear <strong>{patient_name}</strong>,</p>
    <p style="color:#4a4840;font-size:14px">Your insurance claim status has been updated.</p>
    <div style="background:#f9f8f5;border-radius:8px;padding:16px;margin:16px 0;border:1px solid #e4e2db">
      <p style="margin:0 0 8px;font-size:13px;color:#8a8780">Claim ID: <span style="font-family:monospace;color:#1a1916">{claim_id[:8]}…</span></p>
      <p style="margin:0 0 8px;font-size:15px;font-weight:700;color:{color}">
        Status: {new_status.replace('_',' ').upper()}
      </p>
      {f'<p style="font-size:13px;color:#4a4840;margin:6px 0">Remarks: {remarks}</p>' if remarks else ''}
      {f'<p style="font-size:12px;color:#8a8780;margin:4px 0">Updated by: {insurer_org}</p>' if insurer_org else ''}
    </div>
    <p style="color:#4a4840;font-size:13px">Your claim has been {desc}.</p>
    """
    return _send(
        to_emails=[patient_email],
        subject=f"Claim {new_status.replace('_',' ').title()} — CodePerfect",
        html_body=_base_html(title, body),
    )