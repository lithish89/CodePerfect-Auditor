"""
Audit Log Database  —  app/database/audit_db.py
------------------------------------------------
Stores every audit run in a local SQLite database.
Zero setup required — SQLite is built into Python, no server needed.

Tables:
  audit_logs     — one row per /analyze call
  audit_codes    — child rows: every code (matched / missing / extra) per run

Usage:
  from app.database.audit_db import init_db, save_audit, get_all_audits, get_audit_by_id
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Database location ─────────────────────────────────────────────────────
DB_DIR  = Path("app/data")
DB_PATH = DB_DIR / "audit_logs.db"


# ── Schema ────────────────────────────────────────────────────────────────

CREATE_AUDIT_LOGS = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    clinical_note    TEXT NOT NULL,
    risk_level       TEXT NOT NULL,
    revenue_risk     TEXT,
    action_required  TEXT,
    matched_count    INTEGER DEFAULT 0,
    missing_count    INTEGER DEFAULT 0,
    extra_count      INTEGER DEFAULT 0,
    total_ai_icd     INTEGER DEFAULT 0,
    total_ai_cpt     INTEGER DEFAULT 0,
    total_human      INTEGER DEFAULT 0,
    warnings         TEXT,
    recommendations  TEXT,
    entities_json    TEXT,
    status           TEXT DEFAULT 'pending'
)
"""

CREATE_AUDIT_CODES = """
CREATE TABLE IF NOT EXISTS audit_codes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id    TEXT NOT NULL REFERENCES audit_logs(id),
    code        TEXT NOT NULL,
    code_type   TEXT NOT NULL,
    category    TEXT NOT NULL,
    description TEXT,
    confidence  REAL,
    risk        TEXT,
    note        TEXT
)
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_audit_created   ON audit_logs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_audit_risk      ON audit_logs(risk_level)",
    "CREATE INDEX IF NOT EXISTS idx_audit_status    ON audit_logs(status)",
    "CREATE INDEX IF NOT EXISTS idx_codes_audit_id  ON audit_codes(audit_id)",
    "CREATE INDEX IF NOT EXISTS idx_codes_category  ON audit_codes(category)",
]


# ── Connection helper ─────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row       # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Initialisation ────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables and indexes. Safe to call multiple times."""
    with _get_conn() as conn:
        conn.execute(CREATE_AUDIT_LOGS)
        conn.execute(CREATE_AUDIT_CODES)
        for idx in CREATE_INDEXES:
            conn.execute(idx)
    print(f"[AuditDB] Database ready at {DB_PATH}")


# ── Save ──────────────────────────────────────────────────────────────────

def save_audit(
    clinical_note: str,
    entities:      dict,
    ai_codes:      dict,
    audit_result:  Optional[dict],
    human_codes:   list,
) -> str:
    """
    Persist one full audit run and return its UUID.

    Parameters mirror the /analyze endpoint output directly — just pass
    through whatever the agents returned.
    """
    audit_id   = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"

    # ── Flatten audit_result safely ───────────────────────────────────────
    if audit_result:
        summary         = audit_result.get("summary", {})
        risk_level      = audit_result.get("risk_level", "unknown")
        revenue_risk    = audit_result.get("revenue_risk", "")
        action_required = audit_result.get("action_required", "")
        matched_count   = summary.get("matched", 0)
        missing_count   = summary.get("missing_count", 0)
        extra_count     = summary.get("extra_count", 0)
        recommendations = json.dumps(audit_result.get("recommendations", []))
        missing_codes   = audit_result.get("missing_codes", [])
        extra_codes     = audit_result.get("extra_codes", [])
        matched_codes   = audit_result.get("matched", [])
    else:
        risk_level = revenue_risk = action_required = "no_audit"
        matched_count = missing_count = extra_count = 0
        recommendations = "[]"
        missing_codes = extra_codes = matched_codes = []

    warnings       = json.dumps(ai_codes.get("warnings", []))
    total_ai_icd   = len(ai_codes.get("icd_codes", []))
    total_ai_cpt   = len(ai_codes.get("cpt_codes", []))
    total_human    = len(human_codes)
    entities_json  = json.dumps({
        k: v for k, v in entities.items() if k != "raw_text"
    })

    with _get_conn() as conn:
        # ── Insert main log row ───────────────────────────────────────────
        conn.execute("""
            INSERT INTO audit_logs (
                id, created_at, clinical_note, risk_level, revenue_risk,
                action_required, matched_count, missing_count, extra_count,
                total_ai_icd, total_ai_cpt, total_human,
                warnings, recommendations, entities_json, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            audit_id, created_at,
            clinical_note[:2000],   # truncate very long notes
            risk_level, revenue_risk, action_required,
            matched_count, missing_count, extra_count,
            total_ai_icd, total_ai_cpt, total_human,
            warnings, recommendations, entities_json,
            "reviewed" if human_codes else "ai_only",
        ))

        # ── Insert code rows ──────────────────────────────────────────────
        def _insert_codes(code_list: list, category: str) -> None:
            for c in code_list:
                code_val = c.get("icd_code") or c.get("cpt_code") or c.get("code", "")
                conn.execute("""
                    INSERT INTO audit_codes
                        (audit_id, code, code_type, category, description, confidence, risk, note)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    audit_id,
                    code_val,
                    "ICD-10" if "icd_code" in c or c.get("type") == "ICD-10" else "CPT",
                    category,
                    c.get("description", ""),
                    c.get("confidence"),
                    c.get("risk", ""),
                    c.get("note", ""),
                ))

        _insert_codes(ai_codes.get("icd_codes", []),          "ai_confirmed")
        _insert_codes(ai_codes.get("cpt_codes", []),          "ai_confirmed")
        _insert_codes(ai_codes.get("manual_review", []),      "manual_review")
        _insert_codes(matched_codes,                           "matched")
        _insert_codes(missing_codes,                           "missing")
        _insert_codes(extra_codes,                             "extra")
        
        conn.commit()

    return audit_id


# ── Read ──────────────────────────────────────────────────────────────────

def get_all_audits(
    limit:      int = 50,
    offset:     int = 0,
    risk_filter: Optional[str] = None,
) -> list[dict]:
    """Return recent audit log summaries, newest first."""
    query  = "SELECT * FROM audit_logs"
    params: list = []
    if risk_filter:
        query += " WHERE risk_level = ?"
        params.append(risk_filter)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_audit_by_id(audit_id: str) -> Optional[dict]:
    """Return a full audit record including all code rows."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM audit_logs WHERE id = ?", (audit_id,)
        ).fetchone()
        if not row:
            return None
        result = dict(row)

        codes = conn.execute(
            "SELECT * FROM audit_codes WHERE audit_id = ? ORDER BY category, code",
            (audit_id,),
        ).fetchall()
        result["codes"] = [dict(c) for c in codes]

        # Re-parse JSON fields
        result["warnings"]       = json.loads(result.get("warnings") or "[]")
        result["recommendations"]= json.loads(result.get("recommendations") or "[]")
        result["entities"]       = json.loads(result.get("entities_json") or "{}")

    return result


def get_audit_stats() -> dict:
    """Return aggregate statistics across all audit runs."""
    with _get_conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        by_risk   = conn.execute("""
            SELECT risk_level, COUNT(*) as count
            FROM audit_logs GROUP BY risk_level
        """).fetchall()
        avg_miss  = conn.execute(
            "SELECT AVG(missing_count) FROM audit_logs WHERE status='reviewed'"
        ).fetchone()[0]
        avg_extra = conn.execute(
            "SELECT AVG(extra_count) FROM audit_logs WHERE status='reviewed'"
        ).fetchone()[0]
        recent_7d = conn.execute("""
            SELECT COUNT(*) FROM audit_logs
            WHERE created_at >= datetime('now', '-7 days')
        """).fetchone()[0]
        top_missing = conn.execute("""
            SELECT code, description, COUNT(*) as freq
            FROM audit_codes WHERE category='missing'
            GROUP BY code ORDER BY freq DESC LIMIT 10
        """).fetchall()

    return {
        "total_audits":          total,
        "audits_last_7_days":    recent_7d,
        "by_risk_level":         {r["risk_level"]: r["count"] for r in by_risk},
        "avg_missing_per_audit": round(avg_miss or 0, 2),
        "avg_extra_per_audit":   round(avg_extra or 0, 2),
        "top_missing_codes":     [dict(r) for r in top_missing],
    }


def update_audit_status(audit_id: str, status: str) -> bool:
    """Update status field. Valid values: pending, reviewed, approved, disputed."""
    valid = {"pending", "reviewed", "approved", "disputed", "ai_only"}
    if status not in valid:
        return False
    with _get_conn() as conn:
        rows = conn.execute(
            "UPDATE audit_logs SET status=? WHERE id=?", (status, audit_id)
        ).rowcount
    return rows > 0


def delete_audit(audit_id: str) -> bool:
    """Hard-delete an audit log and all its code rows."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM audit_codes WHERE audit_id=?", (audit_id,))
        rows = conn.execute(
            "DELETE FROM audit_logs WHERE id=?", (audit_id,)
        ).rowcount
    return rows > 0