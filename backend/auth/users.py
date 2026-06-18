"""
User store + password hashing (no external dependencies).

Passwords are never stored in plaintext: each is salted and hashed with
PBKDF2-HMAC-SHA256 (200k iterations). Users live in a small SQLite database
(data/auth.db). Includes an admin CLI (see module docstring in __init__).
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
AUTH_DB = Path(os.getenv("AUTH_DB_PATH", str(ROOT / "data" / "auth.db")))

_PBKDF2_ROUNDS = 200_000
_USER_RE = re.compile(r"^[A-Za-z0-9._@-]{3,64}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RESET_TTL_SECONDS = 30 * 60   # password-reset links expire after 30 minutes


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


def valid_dob(dob: str) -> bool:
    """A plausible ISO date of birth (YYYY-MM-DD): a real calendar date that is not in the
    future and sits within a sane range (1900-01-01 .. today)."""
    try:
        d = datetime.date.fromisoformat((dob or "").strip())
    except ValueError:
        return False
    return datetime.date(1900, 1, 1) <= d <= datetime.date.today()


# ----------------------------------------------------------------------
# Password hashing
# ----------------------------------------------------------------------
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, dk_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def valid_user_id(user_id: str) -> bool:
    return bool(_USER_RE.match(user_id or ""))


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            is_admin      INTEGER NOT NULL DEFAULT 0,
            created_at    REAL NOT NULL
        )
    """)
    # Migration: add the optional email column to pre-existing databases.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "email" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "date_of_birth" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN date_of_birth TEXT")
    # One-time, single-use, expiring password-reset tokens (only the hash is stored).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            token_hash TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            expires_at REAL NOT NULL,
            used       INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
    """)
    return conn


def create_user(user_id: str, password: str, is_admin: bool = False,
                email: Optional[str] = None, date_of_birth: Optional[str] = None) -> None:
    user_id = (user_id or "").strip()
    email = (email or "").strip().lower() or None
    dob = (date_of_birth or "").strip() or None
    if not valid_user_id(user_id):
        raise ValueError("user_id must be 3-64 chars: letters, digits, . _ - @")
    if not password or len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    if email and not valid_email(email):
        raise ValueError("please enter a valid email address")
    if dob and not valid_dob(dob):
        raise ValueError("please enter a valid date of birth")
    with _conn() as conn:
        if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
            raise ValueError(f"user {user_id!r} already exists")
        if email and conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise ValueError("that email is already registered")
        conn.execute(
            "INSERT INTO users (user_id, password_hash, is_admin, created_at, email, date_of_birth) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, hash_password(password), 1 if is_admin else 0, time.time(), email, dob),
        )


def verify_user(user_id: str, password: str) -> bool:
    """Constant-time-ish check; always runs a hash to avoid user enumeration."""
    with _conn() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE user_id = ?",
                           ((user_id or "").strip(),)).fetchone()
    stored = row["password_hash"] if row else hash_password("x")  # dummy work if missing
    return bool(row) and check_password(password or "", stored)


def user_exists(user_id: str) -> bool:
    with _conn() as conn:
        return conn.execute("SELECT 1 FROM users WHERE user_id = ?",
                            ((user_id or "").strip(),)).fetchone() is not None


def set_password(user_id: str, password: str) -> None:
    if not password or len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    with _conn() as conn:
        cur = conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                           (hash_password(password), (user_id or "").strip()))
        if cur.rowcount == 0:
            raise ValueError(f"no such user: {user_id!r}")


# ----------------------------------------------------------------------
# Email + password-reset tokens
# ----------------------------------------------------------------------
def get_email(user_id: str) -> Optional[str]:
    with _conn() as conn:
        row = conn.execute("SELECT email FROM users WHERE user_id = ?",
                           ((user_id or "").strip(),)).fetchone()
    return row["email"] if row else None


def get_dob(user_id: str) -> Optional[str]:
    with _conn() as conn:
        row = conn.execute("SELECT date_of_birth FROM users WHERE user_id = ?",
                           ((user_id or "").strip(),)).fetchone()
    return row["date_of_birth"] if row else None


def set_email(user_id: str, email: str) -> None:
    email = (email or "").strip().lower()
    if email and not valid_email(email):
        raise ValueError("please enter a valid email address")
    with _conn() as conn:
        cur = conn.execute("UPDATE users SET email = ? WHERE user_id = ?",
                           (email or None, (user_id or "").strip()))
        if cur.rowcount == 0:
            raise ValueError(f"no such user: {user_id!r}")


def find_user_by_email(email: str) -> Optional[str]:
    email = (email or "").strip().lower()
    if not email:
        return None
    with _conn() as conn:
        row = conn.execute("SELECT user_id FROM users WHERE email = ?", (email,)).fetchone()
    return row["user_id"] if row else None


def resolve_user(identifier: str) -> Optional[str]:
    """Look up a user by user_id OR email. Returns the user_id, or None."""
    ident = (identifier or "").strip()
    if not ident:
        return None
    if user_exists(ident):
        return ident
    return find_user_by_email(ident)


def _hash_token(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def create_reset_token(user_id: str) -> str:
    """Issue a single-use, 30-minute password-reset token for an existing user.
    Only the SHA-256 hash is stored; the raw token goes in the reset link. Any
    prior unused tokens for the user are invalidated."""
    user_id = (user_id or "").strip()
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _conn() as conn:
        conn.execute("UPDATE password_resets SET used = 1 WHERE user_id = ? AND used = 0", (user_id,))
        conn.execute(
            "INSERT INTO password_resets (token_hash, user_id, expires_at, used, created_at) "
            "VALUES (?, ?, ?, 0, ?)",
            (_hash_token(token), user_id, now + _RESET_TTL_SECONDS, now),
        )
    return token


def consume_reset_token(token: str) -> Optional[str]:
    """Validate + burn a reset token. Returns the user_id if valid (exists, not
    expired, not used); marks it used so it can't be replayed. None otherwise."""
    if not token:
        return None
    th = _hash_token(token)
    now = time.time()
    with _conn() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at, used FROM password_resets WHERE token_hash = ?",
            (th,)).fetchone()
        if not row or row["used"] or row["expires_at"] < now:
            return None
        conn.execute("UPDATE password_resets SET used = 1 WHERE token_hash = ?", (th,))
        return row["user_id"]


def delete_user(user_id: str) -> bool:
    with _conn() as conn:
        return conn.execute("DELETE FROM users WHERE user_id = ?",
                           ((user_id or "").strip(),)).rowcount > 0


def list_users() -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute("SELECT user_id, is_admin, created_at FROM users ORDER BY user_id").fetchall()
    return [{"user_id": r["user_id"], "is_admin": bool(r["is_admin"]),
             "created_at": r["created_at"]} for r in rows]


def count_users() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# ----------------------------------------------------------------------
# Admin CLI
# ----------------------------------------------------------------------
def _main() -> int:
    import argparse
    import getpass

    p = argparse.ArgumentParser(description="Manage app user accounts.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="create a user")
    a.add_argument("user_id")
    a.add_argument("--admin", action="store_true")
    sub.add_parser("list", help="list users")
    d = sub.add_parser("delete", help="delete a user"); d.add_argument("user_id")
    w = sub.add_parser("passwd", help="reset a user's password"); w.add_argument("user_id")
    args = p.parse_args()

    if args.cmd == "list":
        users = list_users()
        if not users:
            print("No users yet. Add one:  python -m backend.auth.users add <user_id>")
        for u in users:
            print(f"  {u['user_id']}{'  (admin)' if u['is_admin'] else ''}")
        return 0

    if args.cmd == "delete":
        print("Deleted." if delete_user(args.user_id) else "No such user.")
        return 0

    # add / passwd both need a password
    pw = getpass.getpass("Password: ")
    if getpass.getpass("Confirm:  ") != pw:
        print("Passwords don't match."); return 1
    try:
        if args.cmd == "add":
            create_user(args.user_id, pw, is_admin=getattr(args, "admin", False))
            print(f"Created user {args.user_id!r}.")
        else:
            set_password(args.user_id, pw)
            print(f"Password updated for {args.user_id!r}.")
    except ValueError as exc:
        print(f"Error: {exc}"); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
