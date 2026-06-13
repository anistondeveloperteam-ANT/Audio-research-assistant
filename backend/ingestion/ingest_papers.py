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
# Ingestion parses CUDA-free: Docling layout runs on the CPU and never attempts the GPU (no
# std::bad_alloc, no retries). The reranker/embedder still use the GPU at query time, and embedding
# is a separate ingestion stage — neither is affected. Set DOCLING_DEVICE=cuda to opt back in.
force_cpu_parsing()

PAPERS_DIR = Path("data/papers")


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


def main():
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