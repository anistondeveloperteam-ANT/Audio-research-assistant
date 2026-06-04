import os
import hashlib
from pathlib import Path

import oracledb
from dotenv import load_dotenv
from tqdm import tqdm

from backend.ingestion.pdf_parser import parse_pdf
from backend.ingestion.document_chunker import chunk_parsed_document

load_dotenv()

PAPERS_DIR = Path("data/papers")


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


def insert_chunk(cur, paper_id, chunk_index, chunk):
    cur.execute(
        """
        INSERT INTO chunks (
            paper_id, section_name, chunk_index, chunk_text, chunk_type,
            page_start, page_end, has_equation, has_algorithm,
            has_table, audio_concepts
        )
        VALUES (
            :paper_id, :section_name, :chunk_index, :chunk_text, :chunk_type,
            :page_start, :page_end, :has_equation, :has_algorithm,
            :has_table, :audio_concepts
        )
        """,
        {
            "paper_id": paper_id,
            "section_name": chunk["section"],
            "chunk_index": chunk_index,
            "chunk_text": chunk["text"],
            "chunk_type": chunk["chunk_type"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "has_equation": chunk["has_equation"],
            "has_algorithm": chunk["has_algorithm"],
            "has_table": chunk["has_table"],
            "audio_concepts": ", ".join(chunk["concepts"]),
        },
    )


def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}")
        return

    conn = connect()
    cur = conn.cursor()

    total_new_chunks = 0

    print(f"Found {len(pdfs)} PDFs")

    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
        file_hash = file_sha256(pdf_path)
        existing_id = paper_exists(cur, file_hash)

        if existing_id:
            print(f"Skipping already ingested: {pdf_path.name}")
            continue

        title = infer_title(pdf_path)

        parsed = parse_pdf(pdf_path)
        chunks = chunk_parsed_document(parsed)

        paper_id = insert_paper(
            cur=cur,
            title=title,
            file_path=pdf_path,
            file_name=pdf_path.name,
            file_hash=file_hash,
            page_count=parsed.get("page_count", 0),
        )

        for i, chunk in enumerate(chunks, start=1):
            insert_chunk(cur, paper_id, i, chunk)

        conn.commit()

        total_new_chunks += len(chunks)

        print(
            f"Ingested: {pdf_path.name} | "
            f"parser={parsed.get('parser')} | "
            f"pages={parsed.get('page_count')} | "
            f"chunks={len(chunks)}"
        )

    cur.execute("SELECT COUNT(*) FROM papers")
    paper_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]

    print("\nIngestion summary:")
    print(f"Papers in DB: {paper_count}")
    print(f"Chunks in DB: {chunk_count}")
    print(f"New chunks added this run: {total_new_chunks}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()