"""
Authentication  —  app/auth/auth.py  (v2)
------------------------------------------
Roles:
  hospital    → submit notes, add patients, send reports
  insurer     → view claims, update claim status
  admin       → full access
"""

from __future__ import annotations
import sqlite3, hashlib, os, secrets
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

SECRET_KEY  = os.environ.get("JWT_SECRET", "codeperfect-secret-key-2024")
TOKEN_HOURS = 8
DB_PATH     = Path("app/data/audit_logs.db")


def _hash(pw: str, salt: str = "") -> str:
    if not salt:
        salt = secrets.token_hex(16)
    # Use fewer iterations for development speed (100k is for production)
    iterations = 10_000  # Reduced from 100_000 for faster testing
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), iterations).hex()
    return f"{salt}:{h}"

def _verify(pw: str, stored: str) -> bool:
    try:
        salt, _ = stored.split(":", 1)
        return _hash(pw, salt) == stored
    except Exception:
        return False


def init_auth_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                password     TEXT NOT NULL,
                role         TEXT NOT NULL DEFAULT 'hospital',
                full_name    TEXT,
                organization TEXT,
                email        TEXT,
                created_at   TEXT
            )
        """)
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            defaults = [
                ("admin",     "admin123",    "admin",    "Administrator",  "",                      "admin@codeperfect.ai"),
                ("hospital1", "hospital123", "hospital", "City Hospital",  "City General Hospital", "billing@cityhospital.com"),
                ("insurer1",  "insurer123",  "insurer",  "BlueCross",      "BlueCross BlueShield",  "claims@bluecross.com"),
            ]
            for u, pw, role, name, org, email in defaults:
                c.execute(
                    "INSERT INTO users (username,password,role,full_name,organization,email,created_at) VALUES (?,?,?,?,?,?,?)",
                    (u, _hash(pw), role, name, org, email, datetime.utcnow().isoformat())
                )
            c.commit()
            print("[Auth] Default users created.")


def get_user(username: str) -> Optional[dict]:
    with sqlite3.connect(str(DB_PATH)) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    with sqlite3.connect(str(DB_PATH)) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def create_user(username: str, password: str, role: str,
                full_name: str = "", organization: str = "", email: str = "") -> tuple:
    try:
        with sqlite3.connect(str(DB_PATH)) as c:
            c.execute(
                "INSERT INTO users (username,password,role,full_name,organization,email,created_at) VALUES (?,?,?,?,?,?,?)",
                (username, _hash(password), role, full_name, organization, email, datetime.utcnow().isoformat())
            )
            c.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def _make_token(data: dict) -> str:
    try:
        import jwt
        payload = {**data, "exp": datetime.utcnow() + timedelta(hours=TOKEN_HOURS)}
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    except ImportError:
        import base64, json
        payload = {**data, "exp": (datetime.utcnow() + timedelta(hours=TOKEN_HOURS)).isoformat()}
        return base64.b64encode(json.dumps(payload).encode()).decode()

def _read_token(token: str) -> Optional[dict]:
    try:
        import jwt
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except ImportError:
        import base64, json
        try:
            p = json.loads(base64.b64decode(token.encode()).decode())
            if datetime.utcnow() > datetime.fromisoformat(p.get("exp","2000-01-01")):
                return None
            return p
        except Exception:
            return None
    except Exception:
        return None


def login(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user or not _verify(password, user["password"]):
        return None
    token = _make_token({
        "sub":  user["username"],
        "role": user["role"],
        "name": user["full_name"] or user["username"],
        "org":  user["organization"] or "",
        "email":user["email"] or "",
    })
    return {
        "token":        token,
        "username":     user["username"],
        "role":         user["role"],
        "full_name":    user["full_name"] or user["username"],
        "organization": user["organization"] or "",
        "email":        user["email"] or "",
    }

def verify_token(token: str) -> Optional[dict]:
    return _read_token(token)

def get_current_user(authorization: str = "") -> Optional[dict]:
    if not authorization:
        return None
    return _read_token(authorization.replace("Bearer ", "").strip())