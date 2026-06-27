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
import os
import queue
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

# Parsing runs as a CPU subprocess (Docling is forced to CPU there); embedding runs IN-PROCESS on
# the GPU (reusing the warm model). So the parser working on the next PDF overlaps the GPU embedding
# the previous one. The vector migrate + turbovec stages load NO model, so they stay cheap subprocesses.
PARSE_STAGE = ("Reading & chunking the PDF", "backend.ingestion.ingest_papers")
EMBED_STAGE_LABEL = "Building embeddings"
MIGRATE_STAGE = ("Updating the vector index", "backend.database.vector_migration", [])
TURBOVEC_STAGE = ("Building turbovec cache", "backend.retrieval.turbovec_index", ["build"])


def _subprocess_env() -> dict:
    """Env for ingest subprocesses with stdout/stderr forced to UTF-8. Without this the child writes
    its logs in the Windows console default (cp1252), so a single Unicode char (⚠, →, …) raises
    UnicodeEncodeError mid-print — which previously aborted the WHOLE ingest when a PDF was skipped."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _post_embed_stages():
    """Subprocess stages that run AFTER all embedding: vector migrate, then turbovec if enabled.
    Neither loads an ML model, so a subprocess is safe + memory-cheap here."""
    stages = [MIGRATE_STAGE]
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
        conn = _connect()                            # bounded by tcp_connect_timeout -> fast when offline
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


def _db_connect_timeout() -> float:
    """Seconds to wait for the DB before giving up. Keeps the library/chat snappy when the DB is
    offline (e.g. the Oracle Docker container isn't started) instead of stalling ~4s per call."""
    try:
        return float(os.getenv("ORACLE_CONNECT_TIMEOUT", "3"))
    except (TypeError, ValueError):
        return 3.0


def _connect():
    import oracledb
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN,
                            tcp_connect_timeout=_db_connect_timeout())


def list_papers() -> list:
    """List indexed papers with their chunk counts (newest first)."""
    out = []
    try:
        conn = _connect()
        cur = conn.cursor()
        # Use scalar sub-queries for the per-paper counts instead of GROUP BY:
        # `title` is a CLOB and Oracle cannot GROUP BY a LOB column (ORA-00932),
        # which made the whole list silently fail and return empty.
        cur.execute(
            """
            SELECT p.id, p.title, p.file_name,
                   (SELECT COUNT(*) FROM chunks c WHERE c.paper_id = p.id),
                   (SELECT COUNT(*) FROM chunks c WHERE c.paper_id = p.id AND c.embedding IS NULL)
            FROM papers p
            ORDER BY p.id DESC
            """
        )
        for pid, title, fname, n, n_unembedded in cur.fetchall():
            if hasattr(title, "read"):
                title = title.read()
            n = int(n or 0)
            unembedded = int(n_unembedded or 0)
            out.append({
                "id": int(pid),
                "title": str(title or fname or "Untitled"),
                "file_name": str(fname or ""),
                "chunks": n,
                # Half-done: parsed/chunked but not fully embedded (or no chunks at all).
                "incomplete": (n == 0) or (unembedded > 0),
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


def remove_incomplete(delete_files: bool = True) -> Dict[str, Any]:
    """Remove all HALF-DONE papers (parsed but not fully embedded) so they can be uploaded again.
    By default the PDF is deleted too, so a fresh upload re-ingests cleanly. Drops the retrieval
    caches when anything changed."""
    from backend.ingestion.ingest_papers import remove_incomplete_papers
    removed = remove_incomplete_papers(delete_files=delete_files)
    if removed:
        _clear_retrieval_caches(remove_turbovec_files=True)
    return {"ok": True, "removed": removed, "count": len(removed), "library": library_stats()}


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
_ingest_state: Dict[str, Any] = {"proc": None, "cancelled": False, "filenames": [], "running": False}
_ingest_lock = threading.Lock()


def begin_ingest(filenames) -> bool:
    """Claim the single ingest slot and record which just-uploaded files this run covers (so a cancel
    removes exactly those). Returns False if an ingest is ALREADY running — the caller must NOT start a
    second one. This single-flight guard prevents concurrent runs (e.g. double-clicking 'Add papers')
    from hammering the LLM/DB at once, which previously caused rate-limit storms and crawl-slow ingest.
    stream_ingest() releases the slot when it finishes; cancel_ingest() releases it on cancel."""
    with _ingest_lock:
        if _ingest_state.get("running"):
            return False
        _ingest_state["running"] = True
        _ingest_state["proc"] = None
        _ingest_state["cancelled"] = False
        _ingest_state["filenames"] = [str(f) for f in (filenames or [])]
        return True


def _end_ingest() -> None:
    """Release the single ingest slot."""
    with _ingest_lock:
        _ingest_state["running"] = False
        _ingest_state["proc"] = None


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


def _spawn_parser(ev_q: "queue.Queue") -> threading.Thread:
    """Start the parse subprocess (CPU Docling) in a background thread that pushes events onto
    `ev_q`: ('line', text) for every output line, ('parsed_paper', text) the moment a paper's
    'Ingested:' line appears (so the main loop can embed it while the parser moves to the next PDF),
    and ('parse_done', exit_code) at the end. Reading on a thread keeps the parser's stdout pipe
    drained while the main loop is busy embedding, so the parser never blocks."""
    def _run():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", PARSE_STAGE[1]],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace", env=_subprocess_env(),
            )
        except Exception as exc:
            ev_q.put(("error", f"Could not start {PARSE_STAGE[0]}: {exc}"))
            ev_q.put(("parse_done", 1))
            return
        _register_ingest_proc(proc)
        if proc.stdout is not None:
            for raw in iter(proc.stdout.readline, ""):
                line = raw.rstrip("\n").strip()
                if not line:
                    continue
                ev_q.put(("line", line))
                if line.startswith("Ingested:"):       # one paper finished parsing + committed
                    ev_q.put(("parsed_paper", line))
            proc.stdout.close()
        ev_q.put(("parse_done", proc.wait()))

    t = threading.Thread(target=_run, daemon=True, name="ingest-parser")
    t.start()
    return t


def _embed_pass(embedded_total: list) -> Iterator[Dict[str, Any]]:
    """Run ONE in-process embed pass (GPU, reusing the warm model) over all currently-pending
    chunks, translating the outcome into UI events. Catches failures so embedding errors surface as
    a clean message, never a 500. Yields a 'cancelled'/'error' event when the caller must stop."""
    from backend.ingestion.embed_chunks import embed_pending_chunks
    try:
        res = embed_pending_chunks(should_cancel=_ingest_cancelled)
    except Exception as exc:                            # noqa: BLE001 - surface, don't crash the stream
        yield {"type": "error", "message": f"{EMBED_STAGE_LABEL} failed: {exc}"}
        return
    if res.get("cancelled"):
        yield {"type": "cancelled", "message": "Ingestion cancelled — partial data removed."}
        return
    n = res.get("embedded", 0)
    if n:
        embedded_total[0] += n
        yield {"type": "log", "line": f"Embedded {n} new chunk(s) on the GPU."}


def _run_subprocess_stage(label, module, extra_args, page_warnings) -> Iterator[Any]:
    """Run one model-free stage (vector migrate / turbovec) as a subprocess, yielding {type:...}
    dicts for output and finally a ('__code__', exit_code) tuple for the caller to branch on."""
    yield {"type": "stage", "label": label}
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", module] + list(extra_args),
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=_subprocess_env(),
        )
    except Exception as exc:
        yield {"type": "error", "message": f"Could not start {label}: {exc}"}
        yield ("__code__", 1)
        return
    _register_ingest_proc(proc)
    if proc.stdout is not None:
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            if "NOT indexed" in line:
                page_warnings.append(line)
            elif _is_log_noise(line):
                continue
            else:
                yield {"type": "log", "line": line}
        proc.stdout.close()
    yield ("__code__", proc.wait())


def stream_ingest() -> Iterator[Dict[str, Any]]:
    """Run ingestion, yielding progress events, and ALWAYS release the single-ingest slot when done
    (so a finished/cancelled/crashed run never blocks the next one). Delegates to the impl below."""
    try:
        yield from _stream_ingest_impl()
    finally:
        _end_ingest()


def _stream_ingest_impl() -> Iterator[Dict[str, Any]]:
    """Run ingestion, yielding progress events {type: stage|log|error|cancelled|done}.

    Parsing runs as a CPU subprocess; embedding runs IN-PROCESS on the GPU (reusing the warm model
    — no second model load, so it can't OOM-crash). Each paper is embedded as soon as it finishes
    parsing, so the parser working on the next PDF overlaps the GPU embedding the previous one.
    Migrate + turbovec run afterwards. Cancellation is honoured between events and between batches;
    page-coverage warnings are surfaced in the final 'done' event."""
    if _ingest_cancelled():
        yield {"type": "cancelled", "message": "Ingestion cancelled."}
        return

    page_warnings: list = []
    embedded_total = [0]
    embed_started = [False]
    ev_q: "queue.Queue" = queue.Queue()
    _spawn_parser(ev_q)

    yield {"type": "stage", "label": PARSE_STAGE[0]}

    def _embed_now() -> Iterator[Dict[str, Any]]:
        # Mark the embed stage the first time we actually embed, then run a pass.
        if not embed_started[0]:
            embed_started[0] = True
            yield {"type": "stage", "label": EMBED_STAGE_LABEL}
        for ev in _embed_pass(embedded_total):
            yield ev

    parse_code = 0
    while True:
        if _ingest_cancelled():
            yield {"type": "cancelled", "message": "Ingestion cancelled — partial data removed."}
            return
        try:
            kind, payload = ev_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if kind == "line":
            line = payload
            if "NOT indexed" in line:
                page_warnings.append(line)
            elif _is_log_noise(line):
                pass
            else:
                yield {"type": "log", "line": line}
        elif kind == "parsed_paper":
            # Overlap: embed this just-parsed paper on the GPU while the parser starts the next PDF.
            for ev in _embed_now():
                yield ev
                if ev.get("type") in ("error", "cancelled"):
                    return
        elif kind == "error":
            yield {"type": "error", "message": payload}
            return
        elif kind == "parse_done":
            parse_code = payload
            break

    if _ingest_cancelled():
        yield {"type": "cancelled", "message": "Ingestion cancelled — partial data removed."}
        return
    if parse_code != 0:
        yield {"type": "error", "message": f"{PARSE_STAGE[0]} failed (exit code {parse_code})."}
        return

    # Final embed pass — covers the last parsed paper + anything still pending.
    for ev in _embed_now():
        yield ev
        if ev.get("type") in ("error", "cancelled"):
            return

    # Model-free finishing stages.
    for label, module, extra_args in _post_embed_stages():
        if _ingest_cancelled():
            yield {"type": "cancelled", "message": "Ingestion cancelled — partial data removed."}
            return
        code = None
        for ev in _run_subprocess_stage(label, module, extra_args, page_warnings):
            if isinstance(ev, tuple):
                code = ev[1]
            else:
                yield ev
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


def stream_embed_pending() -> Iterator[Dict[str, Any]]:
    """Finish HALF-DONE papers WITHOUT re-parsing: embed every already-parsed-but-unembedded chunk
    IN-PROCESS on the GPU, then run the model-free migrate + turbovec stages. This is the 'Finish
    embedding' action — it lets the UI complete half-done papers instead of only removing them. Always
    releases the single-ingest slot when done (so a finished/crashed run never blocks the next one)."""
    try:
        yield from _stream_embed_pending_impl()
    finally:
        _end_ingest()


def _stream_embed_pending_impl() -> Iterator[Dict[str, Any]]:
    if _ingest_cancelled():
        yield {"type": "cancelled", "message": "Embedding cancelled."}
        return

    page_warnings: list = []
    embedded_total = [0]
    yield {"type": "stage", "label": EMBED_STAGE_LABEL}
    for ev in _embed_pass(embedded_total):                 # one pass embeds ALL pending chunks
        yield ev
        if ev.get("type") in ("error", "cancelled"):
            return

    if embedded_total[0] == 0:                             # nothing was pending — don't run migrate
        yield {"type": "done", "message": "Nothing to embed — every paper is already indexed.",
               "library": library_stats(), "page_warnings": []}
        return

    for label, module, extra_args in _post_embed_stages():   # vector migrate, then turbovec
        if _ingest_cancelled():
            yield {"type": "cancelled", "message": "Embedding cancelled."}
            return
        code = None
        for ev in _run_subprocess_stage(label, module, extra_args, page_warnings):
            if isinstance(ev, tuple):
                code = ev[1]
            else:
                yield ev
        if code != 0:
            yield {"type": "error", "message": f"{label} failed (exit code {code})."}
            return

    _clear_retrieval_caches()
    yield {"type": "done",
           "message": f"Embedded {embedded_total[0]} chunk(s) — your library is ready.",
           "library": library_stats(), "page_warnings": page_warnings}
