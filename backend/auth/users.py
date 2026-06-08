"""
User store + password hashing (no external dependencies).

Passwords are never stored in plaintext: each is salted and hashed with
PBKDF2-HMAC-SHA256 (200k iterations). Users live in a small SQLite database
(data/auth.db). Includes an admin CLI (see module docstring in __init__).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
AUTH_DB = Path(os.getenv("AUTH_DB_PATH", str(ROOT / "data" / "auth.db")))

_PBKDF2_ROUNDS = 200_000
_USER_RE = re.compile(r"^[A-Za-z0-9._@-]{3,64}$")


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
    return conn


def create_user(user_id: str, password: str, is_admin: bool = False) -> None:
    user_id = (user_id or "").strip()
    if not valid_user_id(user_id):
        raise ValueError("user_id must be 3-64 chars: letters, digits, . _ - @")
    if not password or len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    with _conn() as conn:
        if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
            raise ValueError(f"user {user_id!r} already exists")
        conn.execute(
            "INSERT INTO users (user_id, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
            (user_id, hash_password(password), 1 if is_admin else 0, time.time()),
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
