#!/usr/bin/env python
"""
pipeline.py -- build / refresh the research paper search index.

This runs the full ingestion pipeline end-to-end, in order:

    1. Ingest papers   -- parse every PDF in data/papers/ and chunk it
    2. Embed chunks     -- compute embeddings for all chunks
    3. Vector migrate   -- load embeddings into the Oracle vector index

Usage:
    python pipeline.py                # full rebuild (all stages, every paper)
    python pipeline.py --incremental  # only process PDFs that changed since last run

Each stage runs as `python -m backend.<...>` from the project root, so the
`backend` package imports resolve correctly. Output is streamed live; the
pipeline stops at the first stage that fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (label, module) executed in order for a full rebuild.
FULL_STAGES = [
    ("Ingest papers (parse + chunk)", "backend.ingestion.ingest_papers"),
    ("Embed chunks",                  "backend.ingestion.embed_chunks"),
    ("Migrate into Oracle vector index", "backend.database.vector_migration"),
]

INCREMENTAL_MODULE = "backend.ingestion.incremental_index"


def run_stage(label: str, module: str) -> int:
    """Run one stage as a module, streaming its output. Returns its exit code."""
    print("\n" + "=" * 70)
    print(f">> {label}")
    print(f"   (python -m {module})")
    print("=" * 70, flush=True)
    return subprocess.call([sys.executable, "-m", module], cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the search index.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only process PDFs that changed since the last run.")
    args = parser.parse_args()

    started = time.time()

    if args.incremental:
        stages = [("Incremental index (changed PDFs only)", INCREMENTAL_MODULE)]
    else:
        stages = FULL_STAGES

    for label, module in stages:
        code = run_stage(label, module)
        if code != 0:
            print(f"\nFAILED at stage: {label} (exit code {code}).", file=sys.stderr)
            return code

    elapsed = time.time() - started
    print(f"\nPipeline complete in {elapsed:.0f}s. Index is ready.")
    print("Launch the app with:  python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
