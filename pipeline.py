#!/usr/bin/env python
"""
pipeline.py -- build / refresh / inspect the research-paper search index.

The pipeline turns the PDFs in data/papers/ into a searchable index, in order:

    1. Ingest papers  -- parse every PDF and split it into tagged chunks
    2. Embed chunks    -- turn each chunk into a 768-d embedding vector
    3. Vector migrate  -- load the vectors into the Oracle vector index
    4. Optional turbovec cache -- compressed dense-vector accelerator

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

# (label, module, extra_args) run in order for a full rebuild.
FULL_STAGES = [
    ("Ingest papers (parse + chunk)",    "backend.ingestion.ingest_papers", []),
    ("Embed chunks",                     "backend.ingestion.embed_chunks", []),
    ("Migrate into Oracle vector index", "backend.database.vector_migration", []),
]
INCREMENTAL_MODULE = "backend.ingestion.incremental_index"
TURBOVEC_STAGE = ("Build turbovec vector cache", "backend.retrieval.turbovec_index", ["build"])


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
    try:
        from backend.retrieval.turbovec_index import status as turbovec_status

        tv = turbovec_status()
        state = "valid" if tv.get("valid") else "missing/stale"
        enabled = "enabled" if tv.get("enabled") else "disabled"
        print(f"turbovec cache       : {enabled}, {state}")
    except Exception:
        print("turbovec cache       : unavailable")

    # Compute device — shows whether the GPU (e.g. an RTX 3050) is actually used. The reranker
    # always runs here; embeddings only when EMBEDDING_PROVIDER=local (else they're cloud).
    try:
        import torch
        from backend.common.device import resolve_device
        from backend.common.embeddings import provider as embed_provider
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        embed_dev = resolve_device("EMBEDDING_DEVICE") if embed_provider() == "local" else "cloud"
        print(f"Compute device       : GPU={gpu}; embeddings={embed_dev}, "
              f"reranker={resolve_device('RERANKER_DEVICE')}")
    except Exception:
        pass

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
def turbovec_stage_enabled() -> bool:
    try:
        from backend.retrieval.turbovec_index import build_in_pipeline_enabled

        return build_in_pipeline_enabled()
    except Exception:
        return False


def run_stage(label: str, module: str, extra_args=None) -> int:
    """Run one stage as a module, streaming its output. Returns its exit code."""
    extra_args = list(extra_args or [])
    cmd = [sys.executable, "-m", module] + extra_args
    print("\n" + "=" * 70)
    print(f">> {label}")
    print("   (" + " ".join(["python", "-m", module] + extra_args) + ")")
    print("=" * 70, flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


# ----------------------------------------------------------------------
# Re-index a single paper (repair) — delete its rows, keep the PDF, re-ingest
# ----------------------------------------------------------------------
def _resolve_target(filename: str, names):
    """Resolve a user-supplied name to one PDF: exact match, else a UNIQUE case-insensitive
    substring match. Returns (matched_name | None, error_message | None)."""
    if filename in names:
        return filename, None
    matches = [n for n in names if filename.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"no PDF in data/papers/ matches '{filename}'"
    return None, f"'{filename}' matches {len(matches)} PDFs ({', '.join(matches)}); be more specific"


def _delete_paper_rows(file_name: str) -> int:
    """Delete a paper's indexed rows (concept links, chunks, paper) by file_name. Keeps the PDF.
    Returns how many paper rows were removed."""
    conn = connect_oracle()
    cur = conn.cursor()
    cur.execute("SELECT id FROM papers WHERE file_name = :n", {"n": file_name})
    ids = [r[0] for r in cur.fetchall()]
    for pid in ids:
        try:
            cur.execute("DELETE FROM chunk_concepts WHERE chunk_id IN "
                        "(SELECT id FROM chunks WHERE paper_id = :p)", {"p": pid})
        except Exception:
            pass
        cur.execute("DELETE FROM chunks WHERE paper_id = :p", {"p": pid})
        cur.execute("DELETE FROM papers WHERE id = :p", {"p": pid})
    conn.commit()
    conn.close()
    return len(ids)


def reindex_paper(filename: str) -> int:
    """Re-process ONE paper from scratch: clear its indexed rows (keep the PDF), then run the full
    ingest -> embed -> vector-migrate stages. Other papers stay skipped by content hash, so only
    this one is reprocessed. Use to repair a paper that was indexed with missing pages."""
    names = [p.name for p in PAPERS_DIR.glob("*.pdf")]
    target, err = _resolve_target(filename, names)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    ok, oerr = oracle_reachable()
    if not ok:
        print(f"ERROR: cannot reach Oracle at {ORACLE_DSN}: {oerr}", file=sys.stderr)
        print("       Start it first, e.g.  docker start oracle-ai-db", file=sys.stderr)
        return 1

    removed = _delete_paper_rows(target)
    print(f"Re-indexing '{target}' — cleared {removed} existing paper row(s); the PDF is kept.")

    stages = list(FULL_STAGES)
    if turbovec_stage_enabled():
        stages.append(TURBOVEC_STAGE)

    started = time.time()
    for label, module, extra_args in stages:
        code = run_stage(label, module, extra_args)
        if code != 0:
            print(f"\nFAILED at stage: {label} (exit code {code}).", file=sys.stderr)
            return code
    print(f"\nRe-index complete in {time.time() - started:.0f}s for '{target}'.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, refresh, or inspect the search index.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process PDFs that changed since the last run.")
    parser.add_argument("--status", action="store_true",
                        help="Show what is currently indexed and exit (no rebuild).")
    parser.add_argument("--corpus-report", action="store_true",
                        help="Write a corpus coverage report (papers, chunks, topics, gaps) and exit.")
    parser.add_argument("--inspect-chunks", nargs="?", const="", metavar="PAPER_ID",
                        help="List papers + chunk counts, or dump one paper's chunks. No rebuild.")
    parser.add_argument("--reindex", metavar="FILENAME",
                        help="Re-process ONE paper from scratch (delete its rows, keep the PDF, "
                             "re-ingest) — use to repair a paper indexed with missing pages.")
    parser.add_argument("--remove-incomplete", action="store_true",
                        help="Remove HALF-DONE papers (parsed but not fully embedded) so they can be "
                             "re-ingested cleanly. Add --delete-files to also delete the PDFs.")
    parser.add_argument("--delete-files", action="store_true",
                        help="With --remove-incomplete: also delete the PDF files (then upload again).")
    args = parser.parse_args()

    if args.status:
        return show_status()

    if args.remove_incomplete:
        from backend.ingestion.ingest_papers import remove_incomplete_papers
        removed = remove_incomplete_papers(delete_files=args.delete_files)
        if not removed:
            print("No half-done papers found — nothing to remove.")
        else:
            print(f"Removed {len(removed)} half-done paper(s):")
            for name in removed:
                print(f"  - {name}")
            print("Upload them again" if args.delete_files
                  else "Re-run `python pipeline.py` (or re-upload) to index them cleanly.")
        return 0

    if args.reindex:
        return reindex_paper(args.reindex)

    if args.corpus_report:
        from backend.evaluation.corpus_report import run_report
        run_report()
        return 0

    if args.inspect_chunks is not None:
        from backend.evaluation.corpus_report import inspect
        inspect(int(args.inspect_chunks) if str(args.inspect_chunks).strip() else None)
        return 0

    if not preflight(args.incremental):
        return 1

    stages = (
        [("Incremental index (changed PDFs only)", INCREMENTAL_MODULE, [])]
        if args.incremental else list(FULL_STAGES)
    )
    if turbovec_stage_enabled():
        stages.append(TURBOVEC_STAGE)

    started = time.time()
    for label, module, extra_args in stages:
        code = run_stage(label, module, extra_args)
        if code != 0:
            print(f"\nFAILED at stage: {label} (exit code {code}).", file=sys.stderr)
            return code

    print(f"\nPipeline complete in {time.time() - started:.0f}s. Index is ready.")
    print("Launch the app with:  python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
