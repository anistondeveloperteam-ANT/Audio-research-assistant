"""
Reset the chat history (sessions / turns / facts) and the answer cache to empty — after
backing up each SQLite file first. Your **login** (`data/auth.db`) and your **indexed papers**
(Oracle) are left untouched.

    python -m backend.memory.reset_chats          # back up + wipe chat history (asks first)
    python -m backend.memory.reset_chats --yes     # skip the confirmation

Backups land next to each DB as `<name>.bak-<unix_ts>` and can be restored by copying back.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Which tables to empty in each chat DB (tables absent in a given file are skipped).
CHAT_DBS = {
    "conversations.db": ["turns", "sessions", "facts"],
    "memory.db": ["turns", "sessions", "facts", "answer_cache"],
}


def _backup(db: Path, now_ts: int) -> Path:
    """Online backup (safe even while the app holds the DB open)."""
    dest = db.with_name(f"{db.name}.bak-{now_ts}")
    src = sqlite3.connect(str(db))
    out = sqlite3.connect(str(dest))
    try:
        with out:
            src.backup(out)
    finally:
        src.close()
        out.close()
    return dest


def reset_chat_dbs(data_dir: Path, now_ts: int | None = None) -> List[str]:
    """Back up + empty the chat DBs under data_dir. Returns human-readable summary lines."""
    stamp = int(now_ts if now_ts is not None else time.time())
    lines: List[str] = []
    for name, tables in CHAT_DBS.items():
        db = data_dir / name
        if not db.exists():
            lines.append(f"skip (missing): {name}")
            continue
        bak = _backup(db, stamp)
        conn = sqlite3.connect(str(db), timeout=10.0)
        try:
            with conn:
                for t in tables:
                    try:
                        conn.execute(f"DELETE FROM {t}")
                    except sqlite3.OperationalError:
                        pass   # table not present in this file
            try:
                conn.execute("VACUUM")   # shrink the file; best-effort (may be locked)
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()
        lines.append(f"cleared {name}  (backup: {bak.name})")
    return lines


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--yes" not in args:
        print("This empties your chat history + answer cache (auth.db + Oracle are untouched).")
        print("A timestamped backup of each DB is made first.")
        try:
            if input("Type 'yes' to proceed: ").strip().lower() != "yes":
                print("Aborted.")
                return 1
        except EOFError:
            print("Aborted (no input).")
            return 1
    for line in reset_chat_dbs(DATA):
        print(line)
    print("Done. Restart the app (python run.py) for a clean slate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
