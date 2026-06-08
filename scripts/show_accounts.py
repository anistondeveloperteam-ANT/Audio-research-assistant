"""
Show who is registered and how many conversations each member has.

    python scripts/show_accounts.py

Reads the local databases (data/auth.db, data/memory.db). Passwords are stored
only as salted hashes and are NOT shown — they cannot be recovered, only reset
with:  python -m backend.auth.users passwd <user_id>
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.auth import users as user_store  # noqa: E402
from backend.memory.store import default_db_path  # noqa: E402


def _fmt(ts: float) -> str:
    try:
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def main() -> int:
    # --- Accounts (data/auth.db) ---
    accounts = user_store.list_users()
    print(f"\n=== Registered accounts ({len(accounts)}) — {user_store.AUTH_DB} ===")
    if not accounts:
        print("  (none yet)  add one:  python -m backend.auth.users add <user_id>")
    else:
        print(f"  {'USER ID':<24} {'ADMIN':<7} CREATED")
        print(f"  {'-'*24} {'-'*7} {'-'*16}")
        for u in accounts:
            print(f"  {u['user_id']:<24} {('yes' if u['is_admin'] else 'no'):<7} {_fmt(u['created_at'])}")

    # --- Conversations per user (data/memory.db) ---
    mem_db = default_db_path(ROOT)
    print(f"\n=== Conversations per user — {mem_db} ===")
    if not Path(mem_db).exists():
        print("  (no conversations database yet)")
        return 0
    conn = sqlite3.connect(str(mem_db))
    try:
        rows = conn.execute(
            "SELECT user_id, COUNT(*) AS n, MAX(updated_at) AS last "
            "FROM sessions GROUP BY user_id ORDER BY n DESC"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
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
