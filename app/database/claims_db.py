"""
Patient & Claim Database  —  app/database/claims_db.py
-------------------------------------------------------
Tables:
  patients     → patient demographics linked to audit runs
  claims       → insurance claims raised per audit
  claim_status → status history per claim

Claim statuses: pending | requires_verification | issued | accepted | rejected
"""

from __future__ import annotations
import sqlite3, json, uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

DB_PATH = Path("app/data/audit_logs.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_claims_db() -> None:
    with _conn() as c:
        # Patients table
        c.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id             TEXT PRIMARY KEY,
                audit_id       TEXT,
                name           TEXT NOT NULL,
                phone          TEXT,
                email          TEXT,
                dob            TEXT,
                gender         TEXT,
                address        TEXT,
                insurance_code TEXT,
                insurance_org  TEXT,
                insurer_email  TEXT,
                hospital_name  TEXT,
                created_at     TEXT
            )
        """)
        # Claims table
        c.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id              TEXT PRIMARY KEY,
                audit_id        TEXT NOT NULL,
                patient_id      TEXT,
                patient_name    TEXT,
                patient_email   TEXT,
                patient_phone   TEXT,
                insurance_code  TEXT,
                insurance_org   TEXT,
                insurer_email   TEXT,
                hospital_name   TEXT,
                submitted_by    TEXT,
                icd_codes       TEXT,
                cpt_codes       TEXT,
                risk_level      TEXT,
                revenue_risk    TEXT,
                total_at_risk   TEXT,
                status          TEXT DEFAULT 'pending',
                remarks         TEXT,
                created_at      TEXT,
                updated_at      TEXT,
                report_sent     INTEGER DEFAULT 0
            )
        """)
        # Claim status history
        c.execute("""
            CREATE TABLE IF NOT EXISTS claim_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id   TEXT NOT NULL,
                status     TEXT NOT NULL,
                remarks    TEXT,
                changed_by TEXT,
                changed_at TEXT
            )
        """)
        c.commit()
    print("[ClaimsDB] Tables ready.")


# ── Patients ──────────────────────────────────────────────────────────────────

def save_patient(
    audit_id: str,
    name: str, phone: str, email: str,
    dob: str, gender: str, address: str,
    insurance_code: str, insurance_org: str,
    insurer_email: str, hospital_name: str,
) -> str:
    pid = str(uuid.uuid4())
    with _conn() as c:
        c.execute("""
            INSERT INTO patients
            (id,audit_id,name,phone,email,dob,gender,address,
             insurance_code,insurance_org,insurer_email,hospital_name,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (pid, audit_id, name, phone, email, dob, gender, address,
              insurance_code, insurance_org, insurer_email, hospital_name,
              datetime.utcnow().isoformat()))
        c.commit()
    return pid


def get_patient_by_audit(audit_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM patients WHERE audit_id=?", (audit_id,)).fetchone()
    return dict(row) if row else None


# ── Claims ────────────────────────────────────────────────────────────────────

def raise_claim(
    audit_id: str,
    patient: dict,
    audit_result: dict,
    ai_codes: dict,
    submitted_by: str,
) -> str:
    """Create a new insurance claim from an audit run."""
    claim_id = str(uuid.uuid4())
    now      = datetime.utcnow().isoformat()

    icd_list = [c.get("icd_code","") for c in ai_codes.get("icd_codes", [])]
    cpt_list = [c.get("cpt_code","") for c in ai_codes.get("cpt_codes", [])]

    ar = audit_result or {}
    with _conn() as c:
        c.execute("""
            INSERT INTO claims
            (id,audit_id,patient_id,patient_name,patient_email,patient_phone,
             insurance_code,insurance_org,insurer_email,hospital_name,submitted_by,
             icd_codes,cpt_codes,risk_level,revenue_risk,total_at_risk,
             status,remarks,created_at,updated_at,report_sent)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            claim_id, audit_id,
            patient.get("id",""),
            patient.get("name",""),
            patient.get("email",""),
            patient.get("phone",""),
            patient.get("insurance_code",""),
            patient.get("insurance_org",""),
            patient.get("insurer_email",""),
            patient.get("hospital_name",""),
            submitted_by,
            json.dumps(icd_list),
            json.dumps(cpt_list),
            ar.get("risk_level","unknown"),
            ar.get("revenue_risk",""),
            ar.get("total_at_risk",""),
            "pending", "", now, now, 0
        ))
        # First history entry
        c.execute("""
            INSERT INTO claim_history (claim_id,status,remarks,changed_by,changed_at)
            VALUES (?,?,?,?,?)
        """, (claim_id, "pending", "Claim submitted", submitted_by, now))
        c.commit()
    return claim_id


def get_claim(claim_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        history = c.execute(
            "SELECT * FROM claim_history WHERE claim_id=? ORDER BY changed_at DESC",
            (claim_id,)
        ).fetchall()
        result["history"] = [dict(h) for h in history]
        # Parse JSON arrays
        for f in ("icd_codes","cpt_codes"):
            try:
                result[f] = json.loads(result.get(f,"[]"))
            except Exception:
                result[f] = []
    return result


def get_claims_for_insurer(insurer_email: str) -> list[dict]:
    """Return all claims where insurer_email matches — used for insurer login view."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM claims WHERE insurer_email=? ORDER BY created_at DESC",
            (insurer_email,)
        ).fetchall()
    claims = []
    for row in rows:
        d = dict(row)
        for f in ("icd_codes","cpt_codes"):
            try: d[f] = json.loads(d.get(f,"[]"))
            except Exception: d[f] = []
        claims.append(d)
    return claims


def get_claims_for_hospital(submitted_by: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM claims WHERE submitted_by=? ORDER BY created_at DESC",
            (submitted_by,)
        ).fetchall()
    claims = []
    for row in rows:
        d = dict(row)
        for f in ("icd_codes","cpt_codes"):
            try: d[f] = json.loads(d.get(f,"[]"))
            except Exception: d[f] = []
        claims.append(d)
    return claims


def update_claim_status(
    claim_id: str, status: str,
    remarks: str = "", changed_by: str = ""
) -> bool:
    valid = {"pending","requires_verification","issued","accepted","rejected"}
    if status not in valid:
        return False
    now = datetime.utcnow().isoformat()
    with _conn() as c:
        rows = c.execute(
            "UPDATE claims SET status=?,remarks=?,updated_at=? WHERE id=?",
            (status, remarks, now, claim_id)
        ).rowcount
        if rows:
            c.execute(
                "INSERT INTO claim_history (claim_id,status,remarks,changed_by,changed_at) VALUES (?,?,?,?,?)",
                (claim_id, status, remarks, changed_by, now)
            )
            c.commit()
    return rows > 0


def mark_report_sent(claim_id: str) -> None:
    with _conn() as c:
        c.execute("UPDATE claims SET report_sent=1 WHERE id=?", (claim_id,))
        c.commit()


def get_all_claims(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM claims ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        for f in ("icd_codes","cpt_codes"):
            try: d[f] = json.loads(d.get(f,"[]"))
            except Exception: d[f] = []
        result.append(d)
    return result