"""
Factory reset — wipe ALL local data for a clean, fresh start.

    python -m backend.maintenance.factory_reset            # show what WOULD be removed, then stop
    python -m backend.maintenance.factory_reset --yes      # actually wipe

By default it removes everything that makes the install "used" and leaves a fresh-install state:
  - Oracle index : rows in papers, chunks, concepts, chunk_concepts
  - SQLite DBs   : conversations.db, memory.db, llm_costs.db, auth.db (+ their -wal/-shm sidecars)
  - PDFs         : data/papers/*.pdf
  - Caches       : contents of data/extracted, data/vector_cache, data/external_cache, data/logs
                   (incl. the incremental-index manifest, so the next ingest is a clean rebuild)
  - Backups      : every *.bak* file under data/

It KEEPS committed config (evaluation_questions.json, llm_eval_questions.json) and creates NO
new backups. Spare a category with --keep-oracle / --keep-chats / --keep-costs / --keep-auth /
--keep-pdfs / --keep-caches / --keep-backups.

Stop the web app first (`python run.py`) so the SQLite files aren't locked.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# SQLite databases, grouped so --keep-* can spare a category.
CHAT_DBS = ["conversations.db", "memory.db"]
COST_DBS = ["llm_costs.db"]
AUTH_DBS = ["auth.db"]
# Cache directories whose CONTENTS are cleared (the directory itself is kept).
CACHE_DIRS = ["extracted", "vector_cache", "external_cache", "logs"]
# Committed config that must never be deleted.
KEEP_FILES = {"evaluation_questions.json", "llm_eval_questions.json"}
ORACLE_TABLES = ["chunk_concepts", "concepts", "chunks", "papers"]


def _rm_db(data_dir: Path, name: str, log: List[str]) -> None:
    """Remove a SQLite file and its -wal/-shm sidecars."""
    for suffix in ("", "-wal", "-shm"):
        p = data_dir / (name + suffix)
        if p.exists():
            try:
                p.unlink()
                log.append(f"  removed {p.relative_to(ROOT)}")
            except Exception as exc:
                log.append(f"  ! could not remove {p.name}: {exc} (is the app still running?)")


def wipe_sqlite(data_dir: Path, *, chats=True, costs=True, auth=True) -> List[str]:
    log: List[str] = []
    names = (CHAT_DBS if chats else []) + (COST_DBS if costs else []) + (AUTH_DBS if auth else [])
    for name in names:
        if name in KEEP_FILES:
            continue
        _rm_db(data_dir, name, log)
    return log


def wipe_pdfs(data_dir: Path) -> List[str]:
    log: List[str] = []
    papers = data_dir / "papers"
    for pdf in sorted(papers.glob("*.pdf")) if papers.exists() else []:
        try:
            pdf.unlink()
            log.append(f"  removed {pdf.relative_to(ROOT)}")
        except Exception as exc:
            log.append(f"  ! could not remove {pdf.name}: {exc}")
    return log


def _clear_dir(d: Path, log: List[str]) -> None:
    if not d.exists():
        return
    for child in sorted(d.iterdir()):
        if child.name in KEEP_FILES:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            log.append(f"  cleared {child.relative_to(ROOT)}")
        except Exception as exc:
            log.append(f"  ! could not clear {child.name}: {exc}")


def wipe_caches(data_dir: Path) -> List[str]:
    log: List[str] = []
    for name in CACHE_DIRS:
        _clear_dir(data_dir / name, log)
    return log


def wipe_backups(data_dir: Path) -> List[str]:
    """Remove every *.bak* file anywhere under data/ (e.g. memory.db.bak-1781163228)."""
    log: List[str] = []
    if not data_dir.exists():
        return log
    for p in sorted(data_dir.rglob("*.bak*")):
        if p.is_file():
            try:
                p.unlink()
                log.append(f"  removed {p.relative_to(ROOT)}")
            except Exception as exc:
                log.append(f"  ! could not remove {p.name}: {exc}")
    return log


def wipe_oracle() -> List[str]:
    """Delete all rows from the index tables. Best-effort: a missing/unreachable DB is reported,
    not fatal (the local wipes still run)."""
    log: List[str] = []
    try:
        import oracledb
        from backend.config import ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN
        conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
    except Exception as exc:
        log.append(f"  ! Oracle not cleared (cannot connect): {str(exc).splitlines()[0]}")
        log.append("    start it with `docker start oracle-ai-db` and re-run, if you want it wiped.")
        return log
    cur = conn.cursor()
    for table in ORACLE_TABLES:
        try:
            cur.execute(f"DELETE FROM {table}")
            log.append(f"  cleared Oracle table {table}")
        except Exception as exc:
            log.append(f"  ! could not clear {table}: {str(exc).splitlines()[0]}")
    conn.commit()
    cur.close()
    conn.close()
    return log


def run(data_dir: Path, *, oracle=True, chats=True, costs=True, auth=True,
        pdfs=True, caches=True, backups=True) -> List[str]:
    log: List[str] = []
    if oracle:
        log.append("Oracle index:")
        log += wipe_oracle()
    if chats or costs or auth:
        log.append("SQLite databases:")
        log += wipe_sqlite(data_dir, chats=chats, costs=costs, auth=auth)
    if pdfs:
        log.append("PDF files:")
        log += wipe_pdfs(data_dir)
    if caches:
        log.append("Caches:")
        log += wipe_caches(data_dir)
    if backups:
        log.append("Backups:")
        log += wipe_backups(data_dir)
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description="Wipe all local data for a fresh start.")
    ap.add_argument("--yes", action="store_true", help="Actually perform the wipe.")
    for cat in ("oracle", "chats", "costs", "auth", "pdfs", "caches", "backups"):
        ap.add_argument(f"--keep-{cat}", action="store_true", help=f"Do NOT wipe {cat}.")
    args = ap.parse_args()

    opts = dict(oracle=not args.keep_oracle, chats=not args.keep_chats, costs=not args.keep_costs,
                auth=not args.keep_auth, pdfs=not args.keep_pdfs, caches=not args.keep_caches,
                backups=not args.keep_backups)

    if not args.yes:
        print("DRY RUN — this WILL permanently delete (no backups are made):")
        for k, v in opts.items():
            print(f"  {'WIPE ' if v else 'keep '} {k}")
        print("\nStop the web app first, then re-run with --yes to proceed.")
        return 0

    print("Factory reset — wiping local data (no backups)…\n")
    for line in run(DATA, **opts):
        print(line)
    print("\nDone. Fresh start: re-add PDFs (upload or data/papers/ + `python pipeline.py`),"
          " and sign up again if auth is enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
