"""
scripts/export_memory_cli.py  --  Command-line memory export wrapper

Usage:
    python scripts/export_memory_cli.py [--out OUTPUT_FILE]

Defaults:
    Output to data/exports/research_memory_YYYYMMDD_HHMMSS.tar.gz
    Includes .env (masked) and eval reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make backend importable when running from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.memory.memory_backup import cli_export


def main():
    parser = argparse.ArgumentParser(
        description="Export Research Assistant memory (sessions + facts) to a portable .tar.gz bundle."
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path (default: data/exports/research_memory_<timestamp>.tar.gz)"
    )
    args = parser.parse_args()

    try:
        out_path = cli_export(
            project_root=ROOT,
            output_path=Path(args.out) if args.out else None,
        )
        print(f"OK -- export written to {out_path}")
        sys.exit(0)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Hint: have you run the chat UI at least once to create data/memory.db?")
        sys.exit(2)
    except Exception as exc:
        print(f"\nERROR: export failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
