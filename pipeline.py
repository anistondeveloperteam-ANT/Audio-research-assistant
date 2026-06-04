#!/usr/bin/env python
"""
pipeline.py -- build / refresh / inspect the research-paper search index.

The pipeline turns the PDFs in data/papers/ into a searchable index, in order:

    1. Ingest papers  -- parse every PDF and split it into tagged chunks
    2. Embed chunks    -- turn each chunk into a 768-d embedding vector
    3. Vector migrate  -- load the vectors into the Oracle vector index

Usage:
    python pipeline.py                # full rebuild (all stages, every paper)
    python pipeline.py --incremental  # only process PDFs that changed
    python pipeline.py --status       # show what is currently indexed (no rebuild)

Each stage runs as `python -m backend.<...>` from the project root so the
`backend` package imports resolve. Output streams live; the run stops at the
first stage that fails. A preflight check makes sure the PDFs and the Oracle
database are actually there before doing any work.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import PAPERS_DIR, ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN

# (label, module) run in order for a full rebuild.
FULL_STAGES = [
    ("Ingest papers (parse + chunk)",    "backend.ingestion.ingest_papers"),
    ("Embed chunks",                     "backend.ingestion.embed_chunks"),
    ("Migrate into Oracle vector index", "backend.database.vector_migration"),
]
INCREMENTAL_MODULE = "backend.ingestion.incremental_index"


# ----------------------------------------------------------------------
# Preflight + status helpers
# ----------------------------------------------------------------------
def count_pdfs() -> int:
    return len(list(PAPERS_DIR.glob("*.pdf")))


def connect_oracle():
    """Return an open Oracle connection, or raise with the underlying error."""
    import oracledb
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)


def oracle_reachable() -> tuple[bool, str]:
    try:
        connect_oracle().close()
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0]


def show_status() -> int:
    """Print the current index state without changing anything."""
    print("Index status")
    print("-" * 40)
    print(f"PDFs in data/papers/ : {count_pdfs()}")

    ok, err = oracle_reachable()
    if not ok:
        print("Oracle               : NOT reachable")
        print(f"  reason: {err}")
        print("  fix:    start the database, e.g.  docker start oracle-ai-db")
        return 1

    conn = connect_oracle()
    cur = conn.cursor()

    def count(sql: str):
        try:
            cur.execute(sql)
            return cur.fetchone()[0]
        except Exception:
            return "?"

    papers = count("SELECT COUNT(*) FROM papers")
    chunks = count("SELECT COUNT(*) FROM chunks")
    vectors = count("SELECT COUNT(*) FROM chunks WHERE embedding_vec IS NOT NULL")
    conn.close()

    print(f"Oracle               : reachable ({ORACLE_DSN})")
    print(f"Indexed papers       : {papers}")
    print(f"Indexed chunks       : {chunks}")
    print(f"Chunks with vector   : {vectors}")
    print("\nRun `python pipeline.py` to (re)build, or `python run.py` to use the app.")
    return 0


def preflight(incremental: bool) -> bool:
    """Fail fast with a clear message if prerequisites are missing."""
    n_pdfs = count_pdfs()
    if n_pdfs == 0:
        print(f"ERROR: no PDFs found in {PAPERS_DIR}", file=sys.stderr)
        print("       Add your research papers there, then run again.", file=sys.stderr)
        return False

    ok, err = oracle_reachable()
    if not ok:
        print(f"ERROR: cannot reach Oracle at {ORACLE_DSN}", file=sys.stderr)
        print(f"       {err}", file=sys.stderr)
        print("       Start the database first, e.g.  docker start oracle-ai-db", file=sys.stderr)
        return False

    mode = "incremental (changed PDFs only)" if incremental else "full rebuild (all papers)"
    print(f"Pipeline: {mode}")
    print(f"  PDFs in data/papers/ : {n_pdfs}")
    print(f"  Oracle               : reachable ({ORACLE_DSN})")
    return True


# ----------------------------------------------------------------------
# Stage runner
# ----------------------------------------------------------------------
def run_stage(label: str, module: str) -> int:
    """Run one stage as a module, streaming its output. Returns its exit code."""
    print("\n" + "=" * 70)
    print(f">> {label}")
    print(f"   (python -m {module})")
    print("=" * 70, flush=True)
    return subprocess.call([sys.executable, "-m", module], cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, refresh, or inspect the search index.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process PDFs that changed since the last run.")
    parser.add_argument("--status", action="store_true",
                        help="Show what is currently indexed and exit (no rebuild).")
    args = parser.parse_args()

    if args.status:
        return show_status()

    if not preflight(args.incremental):
        return 1

    stages = (
        [("Incremental index (changed PDFs only)", INCREMENTAL_MODULE)]
        if args.incremental else FULL_STAGES
    )

    started = time.time()
    for label, module in stages:
        code = run_stage(label, module)
        if code != 0:
            print(f"\nFAILED at stage: {label} (exit code {code}).", file=sys.stderr)
            return code

    print(f"\nPipeline complete in {time.time() - started:.0f}s. Index is ready.")
    print("Launch the app with:  python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
