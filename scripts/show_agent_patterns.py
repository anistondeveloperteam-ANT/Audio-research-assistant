"""CLI: show recurring code-agent failure patterns recorded in memory.db (read-only).

Usage:
  python scripts/show_agent_patterns.py
  python scripts/show_agent_patterns.py --days 7 --user local

For a DEVELOPER to read and decide real fixes. The system NEVER changes its own code from this — it is
accumulated evidence, not autonomous behaviour change.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Show recurring code-agent failure patterns (read-only).")
    p.add_argument("--days", type=float, default=0, help="Look-back window in days (0 = all time).")
    p.add_argument("--user", type=str, default=None, help="Restrict to one user_id (default: all).")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass
    from backend.memory.store import MemoryStore, default_db_path

    store = MemoryStore(default_db_path(ROOT))
    max_age = args.days * 86400 if args.days and args.days > 0 else None
    rep = store.agent_failure_patterns(user_id=args.user, max_age_seconds=max_age)

    scope = f"user={args.user}" if args.user else "all users"
    window = f"last {args.days:g} day(s)" if max_age else "all time"
    print(f"CODE-AGENT FAILURE PATTERNS ({scope}, {window})")
    print(f"  runs: {rep['total_runs']}  verified: {rep['verified']}  unverified: {rep['unverified']}")
    print("  (read-only report for the developer - the system does not change its own code)\n")
    if not rep["patterns"]:
        print("  No recurring failure patterns recorded yet.")
        return 0
    for pat in rep["patterns"]:
        print(f"  [{pat['count']:>3}] {pat['pattern']}")
        for ex in pat["examples"][:3]:
            note = f" - {ex['note']}" if ex.get("note") else ""
            print(f"          e.g. {ex['task']}{note}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
