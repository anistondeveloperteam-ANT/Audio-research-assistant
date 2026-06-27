import os
import hashlib
import time
from pathlib import Path

import oracledb
from dotenv import load_dotenv
from tqdm import tqdm

from backend.ingestion.pdf_parser import parse_pdf, force_cpu_parsing
from backend.ingestion.document_chunker import chunk_parsed_document
from backend.ingestion.contextualizer import contextualize_chunks

load_dotenv()

PAPERS_DIR = Path("data/papers")


def _force_utf8_stdio() -> None:
    """Make stdout/stderr emit UTF-8 (replacing un-encodable chars) so Unicode in our logs (→, ⚠, …)
    never raises UnicodeEncodeError on a Windows cp1252 pipe — which otherwise aborts the whole parse
    when a PDF is skipped. Belt-and-suspenders to the PYTHONIOENCODING the web/pipeline spawners set."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def full_document_text(parsed: dict) -> str:
    """The parsed document's text, used as context for the situating sentence."""
    if parsed.get("raw_markdown"):
        return parsed["raw_markdown"]
    return "\n".join(p.get("text", "") for p in parsed.get("pages", []))


def connect():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def infer_title(pdf_path: Path) -> str:
    return pdf_path.stem.replace("_", " ").replace("-", " ").strip()


def paper_exists(cur, file_hash):
    cur.execute("SELECT id FROM papers WHERE file_hash = :hash", {"hash": file_hash})
    row = cur.fetchone()
    return row[0] if row else None


def insert_paper(cur, title, file_path, file_name, file_hash, page_count):
    out_id = cur.var(oracledb.NUMBER)

    cur.execute(
        """
        INSERT INTO papers (title, file_path, file_name, file_hash, page_count)
        VALUES (:title, :file_path, :file_name, :file_hash, :page_count)
        RETURNING id INTO :id
        """,
        {
            "title": title,
            "file_path": str(file_path),
            "file_name": file_name,
            "file_hash": file_hash,
            "page_count": page_count,
            "id": out_id,
        },
    )

    return int(out_id.getvalue()[0])


def insert_chunk(cur, paper_id, chunk_index, chunk, context_text=""):
    cur.execute(
        """
        INSERT INTO chunks (
            paper_id, section_name, chunk_index, chunk_text, context_text, chunk_type,
            page_start, page_end, has_equation, has_algorithm,
            has_table, audio_concepts
        )
        VALUES (
            :paper_id, :section_name, :chunk_index, :chunk_text, :context_text, :chunk_type,
            :page_start, :page_end, :has_equation, :has_algorithm,
            :has_table, :audio_concepts
        )
        """,
        {
            "paper_id": paper_id,
            "section_name": chunk["section"],
            "chunk_index": chunk_index,
            "chunk_text": chunk["text"],
            "context_text": context_text or "",
            "chunk_type": chunk["chunk_type"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "has_equation": chunk["has_equation"],
            "has_algorithm": chunk["has_algorithm"],
            "has_table": chunk["has_table"],
            "audio_concepts": ", ".join(chunk["concepts"]),
        },
    )


def coverage_line(name: str, parsed: dict, n_chunks: int, parse_secs: float = None) -> str:
    """One-line per-PDF ingest report: parser, pages_indexed/pages_total, chunks, and parse time."""
    line = (f"Ingested: {name} | parser={parsed.get('parser')} | "
            f"pages_indexed={parsed.get('pages_indexed')}/{parsed.get('pages_total')} | "
            f"chunks={n_chunks}")
    if parse_secs is not None:
        line += f" | parsed in {parse_secs:.1f}s"
    return line


def coverage_warnings(name: str, parsed: dict) -> list:
    """Per-PDF page-coverage warnings (e.g. 'WARNING: N pages ... NOT indexed'), name-prefixed."""
    return [f"{name}: {w}" for w in parsed.get("warnings", [])]


def find_incomplete_papers(cur):
    """Papers that are HALF-DONE: parsed/chunked but not fully embedded — any chunk with a NULL
    embedding, or a paper with zero chunks. Returns [(id, file_name), ...]."""
    cur.execute(
        "SELECT p.id, p.file_name FROM papers p "
        "WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id AND c.embedding IS NULL) "
        "   OR NOT EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id) "
        "ORDER BY p.id")
    out = []
    for pid, name in cur.fetchall():
        if hasattr(name, "read"):
            name = name.read()
        out.append((int(pid), str(name) if name is not None else ""))
    return out


def purge_paper(cur, paper_id) -> None:
    """Delete one paper and everything derived from it (concept links, chunks, the paper row). The
    caller commits. Mirrors the deletion used elsewhere; there is no ON DELETE CASCADE."""
    try:
        cur.execute("DELETE FROM chunk_concepts WHERE chunk_id IN "
                    "(SELECT id FROM chunks WHERE paper_id = :p)", {"p": paper_id})
    except Exception:
        pass
    cur.execute("DELETE FROM chunks WHERE paper_id = :p", {"p": paper_id})
    cur.execute("DELETE FROM papers WHERE id = :p", {"p": paper_id})


def remove_incomplete_papers(delete_files: bool = False) -> list:
    """Remove half-done papers (parsed but not fully embedded) so they can be re-ingested cleanly,
    and return the list of removed file names. With delete_files=True the PDF is deleted too, so you
    can simply upload it again; otherwise the PDF is kept and the next ingest re-processes it.

    Safe to run when no ingest is in progress: it never touches a FULLY-embedded paper, and does NOT
    run during ingestion (so the embed stage's resume of NULL chunks is preserved)."""
    conn = connect()
    cur = conn.cursor()
    removed = []
    try:
        for pid, name in find_incomplete_papers(cur):
            purge_paper(cur, pid)
            removed.append(name or str(pid))
            if delete_files and name:
                try:
                    (PAPERS_DIR / name).unlink(missing_ok=True)
                except Exception:
                    pass
        conn.commit()
    finally:
        cur.close()
        conn.close()
    return removed


def main():
    _force_utf8_stdio()
    # Hide the GPU from THIS parse process so Docling never attempts it (no std::bad_alloc, no
    # retries). Done inside main() — NOT at import — so merely importing this module (e.g. from a
    # test or the app) never disables the GPU the reranker/embedder use at query time. Embedding is
    # a separate stage. Opt back in with DOCLING_DEVICE=cuda.
    force_cpu_parsing()
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}")
        return

    conn = connect()
    cur = conn.cursor()

    total_new_chunks = 0
    overall_warnings = []

    print(f"Found {len(pdfs)} PDFs")

    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
        # Isolate each PDF: a parse failure or an out-of-memory on ONE document is logged and
        # skipped, so the rest of the batch still indexes instead of the whole run aborting.
        try:
            file_hash = file_sha256(pdf_path)
            existing_id = paper_exists(cur, file_hash)

            if existing_id:
                print(f"Skipping already ingested: {pdf_path.name}")
                continue

            title = infer_title(pdf_path)

            t0 = time.time()
            parsed = parse_pdf(pdf_path)
            parse_secs = time.time() - t0
            chunks = chunk_parsed_document(parsed)

            # Contextual Retrieval: one situating sentence per chunk (cached; "" if disabled/LLM fails).
            contexts = contextualize_chunks(full_document_text(parsed), chunks)

            paper_id = insert_paper(
                cur=cur,
                title=title,
                file_path=pdf_path,
                file_name=pdf_path.name,
                file_hash=file_hash,
                page_count=parsed.get("page_count", 0),
            )

            for i, (chunk, context_text) in enumerate(zip(chunks, contexts), start=1):
                insert_chunk(cur, paper_id, i, chunk, context_text)

            conn.commit()

            total_new_chunks += len(chunks)

            print(coverage_line(pdf_path.name, parsed, len(chunks), parse_secs))
            for w in coverage_warnings(pdf_path.name, parsed):
                print(f"  {w}")
                overall_warnings.append(w)
        except oracledb.IntegrityError:
            # Same content already in the DB (UQ_PAPERS_FILE_HASH) — a duplicate, not an error.
            # (paper_exists() normally skips these first; this also covers a duplicate within the
            # same batch / a row added by a concurrent run.) Skip cleanly; don't warn or abort.
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"Skipping already ingested (duplicate content): {pdf_path.name}")
            continue
        except Exception as exc:        # incl. MemoryError — never abort the batch on one PDF
            try:
                conn.rollback()
            except Exception:
                pass
            msg = (f"Skipped {pdf_path.name}: {type(exc).__name__}: "
                   f"{(str(exc).splitlines() or [''])[0][:160]}")
            print(f"  ⚠ {msg}")
            overall_warnings.append(msg)
            continue

    cur.execute("SELECT COUNT(*) FROM papers")
    paper_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]

    print("\nIngestion summary:")
    print(f"Papers in DB: {paper_count}")
    print(f"Chunks in DB: {chunk_count}")
    print(f"New chunks added this run: {total_new_chunks}")
    if overall_warnings:
        print("\n" + "=" * 64)
        print(f"PAGE COVERAGE WARNINGS ({len(overall_warnings)}) — some pages are NOT indexed:")
        for w in overall_warnings:
            print(f"  - {w}")
        print("=" * 64)
    else:
        print("Page coverage: all pages indexed (no warnings).")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()