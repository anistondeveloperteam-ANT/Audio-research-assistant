"""
Show who is registered and how many conversations each member has.

    python scripts/show_accounts.py

Reads the local databases (data/auth.db, data/memory.db) using only the Python
standard library — no extra packages or virtualenv required. Passwords are stored
only as salted hashes and are NOT shown — they cannot be recovered, only reset
with:  python -m backend.auth.users passwd <user_id>
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH_DB = Path(os.getenv("AUTH_DB_PATH", str(ROOT / "data" / "auth.db")))
MEM_DB = ROOT / "data" / "memory.db"


def _fmt(ts) -> str:
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def _query(db: Path, sql: str):
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def main() -> int:
    # --- Accounts (data/auth.db) ---
    accounts = _query(AUTH_DB, "SELECT user_id, is_admin, created_at FROM users ORDER BY user_id")
    print(f"\n=== Registered accounts — {AUTH_DB} ===")
    if accounts is None:
        print("  (no accounts database yet)  add one:  python -m backend.auth.users add <user_id>")
    elif not accounts:
        print("  (none yet)  add one:  python -m backend.auth.users add <user_id>")
    else:
        print(f"  {'USER ID':<24} {'ADMIN':<7} CREATED")
        print(f"  {'-'*24} {'-'*7} {'-'*16}")
        for user_id, is_admin, created in accounts:
            print(f"  {user_id:<24} {('yes' if is_admin else 'no'):<7} {_fmt(created)}")

    # --- Conversations per user (data/memory.db) ---
    rows = _query(MEM_DB, "SELECT user_id, COUNT(*) AS n, MAX(updated_at) AS last "
                          "FROM sessions GROUP BY user_id ORDER BY n DESC")
    print(f"\n=== Conversations per user — {MEM_DB} ===")
    if rows is None:
        print("  (no conversations database yet)")
    elif not rows:
        print("  (no conversations yet)")
    else:
        print(f"  {'USER ID':<24} {'CHATS':<7} LAST ACTIVITY")
        print(f"  {'-'*24} {'-'*7} {'-'*16}")
        for user_id, n, last in rows:
            print(f"  {user_id:<24} {n:<7} {_fmt(last)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
