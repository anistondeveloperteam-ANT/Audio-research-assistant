"""
Upload + ingestion for the web UI.

- Save an uploaded PDF into data/papers/ (content-hash dedup).
- Stream the 3 ingestion stages (parse+chunk -> embed -> vector-migrate) live.
  Each stage skips work already done, so adding one paper only processes that
  paper. After success the retrieval caches are cleared so the new paper is
  immediately searchable without restarting the server.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import PAPERS_DIR, ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN

BASE_STAGES = [
    ("Reading & chunking the PDF", "backend.ingestion.ingest_papers"),
    ("Building embeddings",         "backend.ingestion.embed_chunks"),
    ("Updating the vector index",   "backend.database.vector_migration"),
]
TURBOVEC_STAGE = ("Building turbovec cache", "backend.retrieval.turbovec_index", ["build"])


def _ingestion_stages():
    stages = [(label, module, []) for label, module in BASE_STAGES]
    try:
        from backend.retrieval.turbovec_index import build_in_pipeline_enabled

        if build_in_pipeline_enabled():
            stages.append(TURBOVEC_STAGE)
    except Exception:
        pass
    return stages


# ----------------------------------------------------------------------
# Saving uploads
# ----------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _existing_by_hash(digest: str) -> str | None:
    if not PAPERS_DIR.exists():
        return None
    for pdf in PAPERS_DIR.glob("*.pdf"):
        try:
            if _sha256(pdf.read_bytes()) == digest:
                return pdf.name
        except Exception:
            continue
    return None


def _safe_target(filename: str) -> Path:
    name = Path(filename).name or "paper.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    target = PAPERS_DIR / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for i in range(1, 1000):
        cand = PAPERS_DIR / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError("Could not find a free filename")


def save_pdf(filename: str, data: bytes) -> Dict[str, Any]:
    """Save bytes as a PDF unless an identical file already exists."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    if not data:
        return {"status": "error", "message": "Empty file."}
    if data[:5] != b"%PDF-":
        return {"status": "error", "message": "That doesn't look like a PDF file."}
    digest = _sha256(data)
    dup = _existing_by_hash(digest)
    if dup:
        return {"status": "duplicate", "filename": dup}
    target = _safe_target(filename)
    target.write_bytes(data)
    return {"status": "saved", "filename": target.name}


def save_pdfs(items: Iterable[Tuple[str, bytes]]) -> Dict[str, Any]:
    """Save MANY uploaded PDFs in one batch. `items` is (filename, data) pairs. Files are saved
    sequentially, so content-hash dedup applies both against the existing library AND within the
    batch (a second identical file in the same batch is reported 'duplicate'). Returns each file's
    result plus saved/duplicate/error counts."""
    results = []
    counts = {"saved": 0, "duplicate": 0, "error": 0}
    for filename, data in items:
        name = filename or "paper.pdf"
        outcome = save_pdf(name, data or b"")
        counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
        results.append({"name": name, **outcome})
    return {"results": results, "total": len(results), **counts}


# ----------------------------------------------------------------------
# Library stats
# ----------------------------------------------------------------------
def library_stats() -> Dict[str, Any]:
    pdfs = len(list(PAPERS_DIR.glob("*.pdf"))) if PAPERS_DIR.exists() else 0
    out: Dict[str, Any] = {"pdfs": pdfs, "papers": None, "chunks": None, "vectors": None}
    try:
        import oracledb
        conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
        cur = conn.cursor()

        def count(sql: str):
            try:
                cur.execute(sql)
                return cur.fetchone()[0]
            except Exception:
                return None

        out["papers"] = count("SELECT COUNT(*) FROM papers")
        out["chunks"] = count("SELECT COUNT(*) FROM chunks")
        out["vectors"] = count("SELECT COUNT(*) FROM chunks WHERE embedding_vec IS NOT NULL")
        conn.close()
    except Exception:
        pass
    return out


def _connect():
    import oracledb
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)


def list_papers() -> list:
    """List indexed papers with their chunk counts (newest first)."""
    out = []
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.id, p.title, p.file_name, COUNT(c.id)
            FROM papers p LEFT JOIN chunks c ON c.paper_id = p.id
            GROUP BY p.id, p.title, p.file_name
            ORDER BY p.id DESC
            """
        )
        for pid, title, fname, n in cur.fetchall():
            if hasattr(title, "read"):
                title = title.read()
            out.append({
                "id": int(pid),
                "title": str(title or fname or "Untitled"),
                "file_name": str(fname or ""),
                "chunks": int(n),
            })
        conn.close()
    except Exception:
        pass
    return out


def delete_paper(paper_id: int) -> Dict[str, Any]:
    """Completely remove a paper and everything derived from it: its chunks (which
    hold the embeddings + native vectors), its concept links, any now-orphaned
    concepts, the papers row, the PDF file, and any cached parse — then drop the
    in-memory retrieval caches so it disappears from search immediately."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT file_name FROM papers WHERE id = :p", {"p": paper_id})
    row = cur.fetchone()
    file_name = row[0] if row else None

    # concept links for this paper's chunks
    try:
        cur.execute(
            "DELETE FROM chunk_concepts WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE paper_id = :p)", {"p": paper_id}
        )
    except Exception:
        pass
    # chunks (this drops the CLOB embeddings AND the native embedding_vec column)
    cur.execute("DELETE FROM chunks WHERE paper_id = :p", {"p": paper_id})
    cur.execute("DELETE FROM papers WHERE id = :p", {"p": paper_id})
    # concepts no longer referenced by any chunk
    try:
        cur.execute("DELETE FROM concepts WHERE id NOT IN "
                    "(SELECT DISTINCT concept_id FROM chunk_concepts)")
    except Exception:
        pass
    conn.commit()
    conn.close()

    # PDF file + any cached parse artifact for it
    if file_name:
        try:
            (PAPERS_DIR / file_name).unlink(missing_ok=True)
        except Exception:
            pass
        _purge_parse_cache(file_name)

    _clear_retrieval_caches(remove_turbovec_files=True)
    return {"ok": True, "deleted": file_name, "library": library_stats()}


def _purge_parse_cache(file_name: str) -> None:
    """Remove any cached parse output for the deleted PDF (best-effort)."""
    try:
        cache_dir = ROOT / "data" / "extracted" / "parser_cache"
        if not cache_dir.exists():
            return
        stem = Path(file_name).stem.lower()
        for f in cache_dir.iterdir():
            if stem in f.name.lower():
                f.unlink(missing_ok=True)
    except Exception:
        pass


def _clear_retrieval_caches(remove_turbovec_files: bool = False) -> None:
    """Drop the BM25/chunk caches so a newly-ingested paper is searchable now."""
    try:
        import backend.retrieval.hybrid_retrieve as hr
        hr._chunks_cache = None
        hr._bm25_cache = None
    except Exception:
        pass
    try:
        import backend.retrieval.turbovec_index as ti
        ti.clear_cache()
        if remove_turbovec_files:
            ti.delete_index_files()
    except Exception:
        pass
    # The local corpus changed, so previously saved answers may now be stale —
    # invalidate the reuse cache so the next question re-searches.
    try:
        from webapp.chat_logic import memory
        memory().clear_answer_cache()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Streaming ingestion
# ----------------------------------------------------------------------
def stream_ingest() -> Iterator[Dict[str, Any]]:
    """Run the 3 ingestion stages, yielding progress events:
    {type: stage|log|error|done}. Page-coverage warnings (pages that failed/were not indexed)
    are collected from the stage output and surfaced in the final 'done' event."""
    page_warnings = []
    for label, module, extra_args in _ingestion_stages():
        yield {"type": "stage", "label": label}
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", module] + list(extra_args),
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            yield {"type": "error", "message": f"Could not start {label}: {exc}"}
            return

        if proc.stdout is not None:
            for raw in iter(proc.stdout.readline, ""):
                line = raw.rstrip("\n")
                if line.strip():
                    if "NOT indexed" in line:
                        page_warnings.append(line.strip())
                    yield {"type": "log", "line": line}
            proc.stdout.close()
        code = proc.wait()
        if code != 0:
            yield {"type": "error", "message": f"{label} failed (exit code {code})."}
            return

    _clear_retrieval_caches()
    message = "Paper indexed and ready."
    if page_warnings:
        message += (f" ⚠ {len(page_warnings)} page-coverage warning(s): some pages were not indexed "
                    "(see the log).")
    yield {"type": "done", "message": message, "library": library_stats(),
           "page_warnings": page_warnings}
