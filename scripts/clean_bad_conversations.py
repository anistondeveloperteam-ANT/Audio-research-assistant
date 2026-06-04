"""
scripts/clean_bad_conversations.py  --  Remove hallucinated/gibberish chats

Scans every conversation in data/memory.db. For each, runs the v8 sanity
check on every USER turn. If ANY user turn in a conversation is gibberish,
the whole conversation is flagged for deletion (because the assistant
likely hallucinated a response to that gibberish, polluting the chat).

ALWAYS asks for confirmation before deleting. ALWAYS backs up memory.db
first to backups/memory_pre_clean_<timestamp>.db.

Usage:
    python scripts/clean_bad_conversations.py            # dry-run preview
    python scripts/clean_bad_conversations.py --apply    # actually delete
    python scripts/clean_bad_conversations.py --all      # nuke EVERYTHING
                                                         # (still asks confirm)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from backend.answering.query_sanity import check_query_sanity
except ImportError:
    print("ERROR: backend.answering.query_sanity not found.")
    print("Run from the project root so the `backend` package is importable.")
    sys.exit(2)


MEMORY_DB = ROOT / "data" / "memory.db"
BACKUPS_DIR = ROOT / "backups"


def find_bad_conversations(db_path: Path):
    """Return list of (session_id, title, n_turns, bad_reasons) for
    every session that contains at least one gibberish user turn."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    bad = []
    try:
        sessions = conn.execute(
            "SELECT id, title FROM sessions ORDER BY created_at DESC"
        ).fetchall()

        for s in sessions:
            user_turns = conn.execute(
                "SELECT content FROM turns "
                "WHERE session_id = ? AND role = 'user' "
                "ORDER BY turn_index ASC",
                (s["id"],)
            ).fetchall()

            if not user_turns:
                continue

            bad_reasons = []
            for t in user_turns:
                content = (t["content"] or "").strip()
                r = check_query_sanity(content)
                if not r.ok:
                    snippet = content[:30] + ("..." if len(content) > 30 else "")
                    bad_reasons.append(f"{snippet!r} -> {r.reason}")

            if bad_reasons:
                total_turns = conn.execute(
                    "SELECT COUNT(*) FROM turns WHERE session_id = ?",
                    (s["id"],)
                ).fetchone()[0]
                bad.append({
                    "id": s["id"],
                    "title": s["title"],
                    "n_turns": total_turns,
                    "bad_reasons": bad_reasons,
                })
    finally:
        conn.close()

    return bad


def list_all_conversations(db_path: Path):
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT s.id, s.title, COUNT(t.id) AS n_turns "
            "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id "
            "GROUP BY s.id ORDER BY s.created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def backup_db(db_path: Path) -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bp = BACKUPS_DIR / f"memory_pre_clean_{ts}.db"
    shutil.copy2(db_path, bp)
    return bp


def delete_sessions(db_path: Path, session_ids):
    """Delete sessions (cascade removes turns + facts via foreign key)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for sid in session_ids:
            conn.execute("DELETE FROM facts WHERE session_id = ?", (sid,))
            conn.execute("DELETE FROM turns WHERE session_id = ?", (sid,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Delete conversations containing gibberish (hallucination-prone)."
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete (default: dry-run preview).")
    parser.add_argument("--all", action="store_true",
                        help="Wipe ALL conversations, not just bad ones.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt (unattended use).")
    args = parser.parse_args()

    if not MEMORY_DB.exists():
        print(f"No memory.db found at {MEMORY_DB} -- nothing to clean.")
        sys.exit(0)

    print("=" * 60)
    if args.all:
        print("CLEAN BAD CONVERSATIONS -- mode: DELETE ALL")
    elif args.apply:
        print("CLEAN BAD CONVERSATIONS -- mode: APPLY")
    else:
        print("CLEAN BAD CONVERSATIONS -- mode: DRY-RUN (preview only)")
    print("=" * 60)
    print()

    if args.all:
        all_chats = list_all_conversations(MEMORY_DB)
        if not all_chats:
            print("No conversations to delete.")
            sys.exit(0)
        print(f"Will delete ALL {len(all_chats)} conversations:")
        for c in all_chats:
            print(f"  - {c['title'][:55]:<55}  ({c['n_turns']} turns)")
        targets = [c["id"] for c in all_chats]
    else:
        bad = find_bad_conversations(MEMORY_DB)
        if not bad:
            print("No bad conversations found. Nothing to clean.")
            sys.exit(0)
        print(f"Found {len(bad)} conversation(s) with gibberish user turns:")
        for b in bad:
            print()
            print(f"  Title:  {b['title'][:60]}")
            print(f"  Turns:  {b['n_turns']}")
            print("  Gibberish detected in:")
            for reason in b["bad_reasons"][:3]:
                print(f"    - {reason}")
            if len(b["bad_reasons"]) > 3:
                print(f"    - (+{len(b['bad_reasons']) - 3} more)")
        targets = [b["id"] for b in bad]

    print()
    if not args.apply and not args.all:
        print("(dry-run -- pass --apply to actually delete these)")
        sys.exit(0)
    if args.all and not args.apply:
        print("(dry-run -- pass --apply with --all to actually delete ALL)")
        sys.exit(0)

    # Confirmation
    if not args.yes:
        print()
        print("=" * 60)
        if args.all:
            print("WARNING: This will delete ALL conversations.")
        else:
            print(f"WARNING: This will delete {len(targets)} conversation(s).")
        print("Your existing memory.db will be backed up first.")
        print("=" * 60)
        resp = input("Type YES to continue: ").strip()
        if resp != "YES":
            print("Aborted. Nothing was changed.")
            sys.exit(3)

    backup_path = backup_db(MEMORY_DB)
    print()
    print(f"Backed up to: {backup_path}")

    delete_sessions(MEMORY_DB, targets)
    print(f"Deleted {len(targets)} conversation(s).")
    print()
    print("Done. Refresh the chat UI to see the changes.")
    sys.exit(0)


if __name__ == "__main__":
    main()
