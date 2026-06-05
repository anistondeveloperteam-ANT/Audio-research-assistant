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
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import PAPERS_DIR, ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN

STAGES = [
    ("Reading & chunking the PDF", "backend.ingestion.ingest_papers"),
    ("Building embeddings",         "backend.ingestion.embed_chunks"),
    ("Updating the vector index",   "backend.database.vector_migration"),
]


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


def _clear_retrieval_caches() -> None:
    """Drop the BM25/chunk caches so a newly-ingested paper is searchable now."""
    try:
        import backend.retrieval.hybrid_retrieve as hr
        hr._chunks_cache = None
        hr._bm25_cache = None
    except Exception:
        pass


# ----------------------------------------------------------------------
# Streaming ingestion
# ----------------------------------------------------------------------
def stream_ingest() -> Iterator[Dict[str, Any]]:
    """Run the 3 ingestion stages, yielding progress events:
    {type: stage|log|error|done}."""
    for label, module in STAGES:
        yield {"type": "stage", "label": label}
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", module],
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
                    yield {"type": "log", "line": line}
            proc.stdout.close()
        code = proc.wait()
        if code != 0:
            yield {"type": "error", "message": f"{label} failed (exit code {code})."}
            return

    _clear_retrieval_caches()
    yield {"type": "done", "message": "Paper indexed and ready.", "library": library_stats()}
