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
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

# Subprocess output that is noise in the UI: tqdm bars, dedup skips, Docling's recovered-page
# failures / tracebacks, and internal device logs. Pages are never lost (PyMuPDF recovers them),
# so these add nothing for the user — the modal should stay a clean checklist.
_LOG_NOISE = re.compile(
    r"\d+%\|"                       # tqdm progress bar
    r"|\bit/s\]"                    # tqdm rate
    r"|^Skipping already ingested"  # content-hash dedup skips
    r"|Stage \w+ failed"            # Docling per-page stage failure (recovered)
    r"|std::bad_alloc"              # native OOM noise (recovered)
    r"|^Traceback "                 # python traceback header
    r"|^File \""                    # traceback frame (stripped line)
    r"|Docling layout device"       # internal device log
)


def _is_log_noise(line: str) -> bool:
    return (not line.strip()) or bool(_LOG_NOISE.search(line))

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
# ----------------------------------------------------------------------
# In-flight ingest tracking (so the UI ✕ can cancel + clean up). Single concurrent
# ingest is assumed (a personal app); guarded by a lock for the cancel cross-thread call.
# ----------------------------------------------------------------------
_ingest_state: Dict[str, Any] = {"proc": None, "cancelled": False, "filenames": []}
_ingest_lock = threading.Lock()


def begin_ingest(filenames) -> None:
    """Mark the start of an ingest run and which just-uploaded files it covers, so a cancel removes
    exactly those (and nothing else)."""
    with _ingest_lock:
        _ingest_state["proc"] = None
        _ingest_state["cancelled"] = False
        _ingest_state["filenames"] = [str(f) for f in (filenames or [])]


def _register_ingest_proc(proc) -> None:
    with _ingest_lock:
        _ingest_state["proc"] = proc


def _ingest_cancelled() -> bool:
    with _ingest_lock:
        return bool(_ingest_state["cancelled"])


def _delete_rows_by_filename(file_name: str) -> int:
    """Delete indexed rows (chunk_concepts, chunks, papers) for ONE paper by file_name. Returns the
    paper rows removed. Best-effort: if Oracle is unreachable there's nothing committed to clean."""
    try:
        conn = _connect()
    except Exception:
        return 0
    try:
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
        cur.close()
        conn.close()
        return len(ids)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return 0


def cancel_ingest() -> Dict[str, Any]:
    """Stop the in-flight ingest: terminate its subprocess and remove ONLY the in-progress papers'
    data — their PDF files + any rows already inserted. Other papers are untouched."""
    with _ingest_lock:
        _ingest_state["cancelled"] = True
        proc = _ingest_state["proc"]
        filenames = list(_ingest_state["filenames"])
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        except Exception:
            pass
    removed = []
    for name in filenames:
        _delete_rows_by_filename(name)
        try:
            p = PAPERS_DIR / name
            if p.exists():
                p.unlink()
        except Exception:
            pass
        removed.append(name)
    _clear_retrieval_caches(remove_turbovec_files=True)
    return {"cancelled": True, "removed": removed}


def stream_ingest() -> Iterator[Dict[str, Any]]:
    """Run the ingestion stages, yielding progress events {type: stage|log|error|cancelled|done}.
    Registers each stage subprocess so a concurrent cancel_ingest() can terminate it; checks the
    cancel flag between stages. Page-coverage warnings are surfaced in the final 'done' event."""
    page_warnings = []
    for label, module, extra_args in _ingestion_stages():
        if _ingest_cancelled():
            yield {"type": "cancelled", "message": "Ingestion cancelled."}
            return
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
        _register_ingest_proc(proc)

        if proc.stdout is not None:
            for raw in iter(proc.stdout.readline, ""):
                line = raw.rstrip("\n").strip()
                if not line:
                    continue
                if "NOT indexed" in line:           # always surface coverage warnings
                    page_warnings.append(line)
                elif _is_log_noise(line):           # drop tqdm/skips/recovered-page traces
                    continue
                yield {"type": "log", "line": line}
            proc.stdout.close()
        code = proc.wait()
        if _ingest_cancelled():
            yield {"type": "cancelled", "message": "Ingestion cancelled — partial data removed."}
            return
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
