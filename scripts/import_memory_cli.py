"""
scripts/import_memory_cli.py  --  Command-line memory import wrapper

Usage:
    python scripts/import_memory_cli.py BUNDLE.tar.gz [--mode merge|replace] [--dry-run]

Examples:
    # Preview what would happen (no changes)
    python scripts/import_memory_cli.py backup.tar.gz --dry-run

    # Merge backup into existing memory (keep existing, add new)
    python scripts/import_memory_cli.py backup.tar.gz --mode merge

    # Wipe existing memory and restore from backup (DESTRUCTIVE)
    python scripts/import_memory_cli.py backup.tar.gz --mode replace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.memory.memory_backup import cli_import


def main():
    parser = argparse.ArgumentParser(
        description="Import a Research Assistant memory bundle into this installation."
    )
    parser.add_argument(
        "bundle",
        help="Path to the .tar.gz export bundle"
    )
    parser.add_argument(
        "--mode", choices=["merge", "replace"], default="merge",
        help="merge (default): keep existing + add new. replace: wipe first."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying the database."
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt (for unattended use)."
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"ERROR: bundle not found: {bundle_path}", file=sys.stderr)
        sys.exit(2)

    # Confirmation for destructive replace mode
    if args.mode == "replace" and not args.dry_run and not args.yes:
        print()
        print("=" * 60)
        print("WARNING: REPLACE mode wipes ALL existing memory first.")
        print("Your existing memory.db will be backed up automatically.")
        print("=" * 60)
        resp = input("Type YES to continue: ").strip()
        if resp != "YES":
            print("Aborted.")
            sys.exit(3)

    try:
        cli_import(
            project_root=ROOT,
            bundle_path=bundle_path,
            mode=args.mode,
            dry_run=args.dry_run,
        )
        print("OK")
        sys.exit(0)
    except ValueError as exc:
        print(f"\nERROR: bundle invalid -- {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\nERROR: import failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
