"""
show_my_data.py  --  read-only viewer v2

Shows everything your system has stored, in one scrollable pass:

  1.  PDF files on disk          (data/papers/)
  2.  Papers table               (Oracle)
  3.  Chunks table               (counts, embedding coverage)
  4.  Chunk-type distribution
  5.  Top sections
  6.  Top audio concepts
  7.  Sample chunks
  8.  Memory database            (data/memory.db -- NEW in v2)
        - Sessions (conversations)
        - Turn counts per session
        - Long-term facts
        - Recent activity
  9.  Extracted reports          (data/extracted/)
  10. Backup folders             (backups/)
  11. Summary

READ-ONLY. Does not modify any data anywhere.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path.cwd()


def read_lob(value):
    if value is None:
        return ""
    if hasattr(value, "read"):
        try:
            return value.read()
        except Exception:
            return ""
    return value


def banner(text: str, char: str = "=") -> None:
    print()
    print(char * 72)
    print(text)
    print(char * 72)


def section(text: str) -> None:
    print()
    print("-" * 72)
    print(text)
    print("-" * 72)


def humanize_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} bytes"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def humanize_time(epoch: float) -> str:
    try:
        return time.strftime("%b %d %Y, %H:%M", time.localtime(epoch))
    except Exception:
        return "?"


# ----------------------------------------------------------------------
# 0. Project root
# ----------------------------------------------------------------------
banner("AUDIO RESEARCH PAPER ASSISTANT -- DATA VIEWER v2 (READ-ONLY)")
print(f"Project root: {ROOT}")


# ----------------------------------------------------------------------
# 1. PDFs
# ----------------------------------------------------------------------
section("1. PDF FILES ON DISK")
pdf_dir = ROOT / "data" / "papers"
pdfs = []
if pdf_dir.exists():
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Location: {pdf_dir}")
    print(f"Total PDFs: {len(pdfs)}")
    if pdfs:
        print("\nFirst 10 PDFs:")
        for p in pdfs[:10]:
            try:
                size = humanize_bytes(p.stat().st_size)
            except Exception:
                size = "?"
            print(f"   {p.name:60} {size}")
        if len(pdfs) > 10:
            print(f"   ... and {len(pdfs) - 10} more")
else:
    print(f"NOTE: {pdf_dir} not found")


# ----------------------------------------------------------------------
# 2-7. Oracle
# ----------------------------------------------------------------------
oracle_ok = False
paper_count = 0
chunk_count = 0
clob_count = 0
vec_count = 0

try:
    import oracledb
    have_oracledb = True
except ImportError:
    have_oracledb = False

if not have_oracledb:
    section("2. ORACLE DATABASE")
    print("oracledb not installed in this Python environment.")
    print("Sections 2-7 will be skipped.")
else:
    section("2. PAPERS TABLE (Oracle)")
    try:
        conn = oracledb.connect(
            user=os.getenv("ORACLE_USER"),
            password=os.getenv("ORACLE_PASSWORD"),
            dsn=os.getenv("ORACLE_DSN"),
        )
        cur = conn.cursor()
        oracle_ok = True

        cur.execute("SELECT COUNT(*) FROM papers")
        paper_count = cur.fetchone()[0]
        print(f"Total papers: {paper_count}")

        if paper_count > 0:
            cur.execute("""
                SELECT p.id, p.title, COUNT(c.id) AS chunk_count
                FROM papers p
                LEFT JOIN chunks c ON c.paper_id = p.id
                GROUP BY p.id, p.title
                ORDER BY p.id
                FETCH FIRST 15 ROWS ONLY
            """)
            rows = cur.fetchall()
            print("\nFirst 15 papers (id | chunks | title):")
            for pid, title, cnt in rows:
                title = str(read_lob(title)) or "(no title)"
                ts = title[:55] + ("..." if len(title) > 55 else "")
                print(f"   [{pid:>3}]  {cnt:>4} chunks  |  {ts}")
            if paper_count > 15:
                print(f"   ... and {paper_count - 15} more")
    except Exception as exc:
        print(f"ERROR connecting to Oracle: {exc}")
        print("Check that Oracle is running and .env credentials are correct.")
        oracle_ok = False


if oracle_ok:
    section("3. CHUNKS TABLE (Oracle)")
    try:
        cur.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        clob_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding_vec IS NOT NULL")
        vec_count = cur.fetchone()[0]
        print(f"Total chunks:             {chunk_count}")
        print(f"With CLOB embedding:      {clob_count}/{chunk_count}")
        print(f"With native VECTOR:       {vec_count}/{chunk_count}")
        if chunk_count > 0:
            print(f"Vector coverage:          {vec_count / chunk_count * 100:.1f}%")
    except Exception as exc:
        print(f"WARN: {exc}")

    section("4. CHUNK-TYPE DISTRIBUTION")
    try:
        cur.execute("""
            SELECT chunk_type, COUNT(*)
            FROM chunks
            GROUP BY chunk_type
            ORDER BY COUNT(*) DESC
        """)
        rows = cur.fetchall()
        if rows:
            print(f"{'chunk_type':30}  count")
            print("   " + "-" * 50)
            for ct, cnt in rows:
                ct_str = str(read_lob(ct)) if ct else "(null)"
                print(f"   {ct_str:30}  {cnt}")
    except Exception as exc:
        print(f"WARN: {exc}")

    section("5. TOP 15 SECTIONS")
    try:
        cur.execute("""
            SELECT section_name, COUNT(*)
            FROM chunks
            GROUP BY section_name
            ORDER BY COUNT(*) DESC
            FETCH FIRST 15 ROWS ONLY
        """)
        rows = cur.fetchall()
        if rows:
            print(f"{'section_name':45}  count")
            print("   " + "-" * 60)
            for sn, cnt in rows:
                sn_str = str(read_lob(sn)) if sn else "(null)"
                ss = sn_str[:43] + ("..." if len(sn_str) > 43 else "")
                print(f"   {ss:45}  {cnt}")
    except Exception as exc:
        print(f"WARN: {exc}")

    section("6. TOP 20 AUDIO CONCEPTS")
    try:
        cur.execute("SELECT audio_concepts FROM chunks WHERE audio_concepts IS NOT NULL")
        counter: Counter = Counter()
        for (raw,) in cur.fetchall():
            txt = str(read_lob(raw))
            if not txt:
                continue
            txt = txt.replace("[", "").replace("]", "")
            for ch in ['"', "'", "\\"]:
                txt = txt.replace(ch, "")
            for piece in txt.split(","):
                piece = piece.strip()
                if piece and 1 < len(piece) < 60:
                    counter[piece] += 1
        if counter:
            print(f"{'concept':32}  count")
            print("   " + "-" * 48)
            for concept, cnt in counter.most_common(20):
                print(f"   {concept:32}  {cnt}")
        else:
            print("   (no audio_concepts populated)")
    except Exception as exc:
        print(f"WARN: {exc}")

    section("7. SAMPLE CHUNKS (3 actual rows)")
    try:
        cur.execute("""
            SELECT c.id, p.title, c.section_name, c.chunk_type,
                   c.page_start, c.page_end, c.audio_concepts, c.chunk_text
            FROM chunks c
            JOIN papers p ON p.id = c.paper_id
            ORDER BY c.id
            FETCH FIRST 3 ROWS ONLY
        """)
        for row in cur.fetchall():
            cid, title, sec, ctype, ps, pe, concepts, text = row
            title = str(read_lob(title))
            sec = str(read_lob(sec))
            ctype = str(read_lob(ctype))
            concepts = str(read_lob(concepts))
            text = str(read_lob(text))
            print()
            print(f"   Chunk #{cid}")
            print(f"   Paper:    {title[:60]}")
            print(f"   Section:  {sec[:50]}")
            print(f"   Type:     {ctype}")
            print(f"   Pages:    {ps}-{pe}")
            print(f"   Concepts: {concepts[:80]}")
            print(f"   Text:     {text[:400]}{'...' if len(text) > 400 else ''}")
    except Exception as exc:
        print(f"WARN: {exc}")

    try:
        cur.close()
        conn.close()
    except Exception:
        pass


# ----------------------------------------------------------------------
# 8. MEMORY DATABASE  (SQLite, new in v2)
# ----------------------------------------------------------------------
section("8. MEMORY DATABASE  (data/memory.db -- conversations + facts)")

mem_path = ROOT / "data" / "memory.db"
if not mem_path.exists():
    print(f"NOTE: {mem_path} not found. Memory layer is created on the first")
    print("      run of RUN_CHAT_UI.bat (after Batch 9 is installed).")
else:
    print(f"Location: {mem_path}")
    print(f"File size: {humanize_bytes(mem_path.stat().st_size)}")

    import sqlite3
    try:
        conn = sqlite3.connect(str(mem_path))
        conn.row_factory = sqlite3.Row

        # Schema version
        try:
            ver = conn.execute("PRAGMA user_version").fetchone()[0]
            print(f"Schema version: {ver}")
        except Exception:
            ver = "?"

        # Counts
        n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        n_turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        n_global = conn.execute("SELECT COUNT(*) FROM facts WHERE scope='global'").fetchone()[0]
        n_session = conn.execute("SELECT COUNT(*) FROM facts WHERE scope='session'").fetchone()[0]
        print(f"\nCounts:")
        print(f"   Conversations (sessions):  {n_sessions}")
        print(f"   Total turns (messages):    {n_turns}")
        print(f"   Long-term facts (global):  {n_global}")
        print(f"   Session-scoped facts:      {n_session}")

        # ----- Sessions list -----
        if n_sessions > 0:
            print(f"\nMost recent 10 conversations:")
            rows = conn.execute("""
                SELECT s.id, s.title, s.updated_at,
                       (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS n_turns
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT 10
            """).fetchall()
            print(f"   {'updated':<22}  {'turns':>5}  title")
            print("   " + "-" * 64)
            for r in rows:
                title = (r["title"] or "(untitled)")[:42]
                print(f"   {humanize_time(r['updated_at']):<22}  {r['n_turns']:>5}  {title}")

        # ----- Long-term facts -----
        if n_global > 0:
            print(f"\nLong-term facts (used by the LLM in every conversation):")
            rows = conn.execute("""
                SELECT key, value, updated_at
                FROM facts WHERE scope='global'
                ORDER BY updated_at DESC
            """).fetchall()
            for r in rows:
                val = (r["value"] or "")[:180]
                print(f"   * {r['key']}: {val}")
        else:
            print("\nNo long-term facts yet.")
            print("   Tip: open the chat UI sidebar and add a few. Example:")
            print("        Key: research_focus")
            print("        Value: low-latency MVDR for embedded systems")

        # ----- Latest 3 turns from the most recent session -----
        if n_sessions > 0 and n_turns > 0:
            sid_row = conn.execute("""
                SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1
            """).fetchone()
            if sid_row:
                latest_turns = conn.execute("""
                    SELECT turn_index, role, content, created_at
                    FROM turns WHERE session_id = ?
                    ORDER BY turn_index DESC LIMIT 3
                """, (sid_row["id"],)).fetchall()
                if latest_turns:
                    print(f"\nLast 3 messages from the most recent conversation:")
                    for r in reversed(latest_turns):  # show in chronological order
                        content = (r["content"] or "").replace("\n", " ")[:250]
                        print(f"   [{r['turn_index']:>2}] {r['role']:>9}: {content}")

        conn.close()
    except Exception as exc:
        print(f"\nERROR reading memory database: {exc}")


# ----------------------------------------------------------------------
# 9. Extracted folder
# ----------------------------------------------------------------------
section("9. EXTRACTED REPORTS & CACHE  (data/extracted/)")
extracted = ROOT / "data" / "extracted"
if extracted.exists():
    files = sorted(extracted.iterdir())
    print(f"Location: {extracted}")
    print(f"Items:    {len(files)}")
    print()
    for f in files:
        try:
            if f.is_file():
                print(f"   [FILE]  {f.name:50}  {humanize_bytes(f.stat().st_size)}")
            elif f.is_dir():
                sub_count = sum(1 for _ in f.iterdir())
                print(f"   [DIR ]  {f.name:50}  {sub_count} items inside")
        except Exception:
            pass
else:
    print(f"NOTE: {extracted} not found")


# ----------------------------------------------------------------------
# 10. Backups
# ----------------------------------------------------------------------
section("10. ROLLBACK BACKUPS  (backups/)")
backups = ROOT / "backups"
if backups.exists():
    folders = sorted([f for f in backups.iterdir() if f.is_dir()])
    print(f"Location: {backups}")
    print(f"Backup folders: {len(folders)}")
    if folders:
        print("\nLast 10:")
        for f in folders[-10:]:
            try:
                contents = list(f.iterdir())
                print(f"   {f.name:55}  {len(contents)} files")
            except Exception:
                pass
else:
    print("No backups folder yet.")


# ----------------------------------------------------------------------
# 11. Summary
# ----------------------------------------------------------------------
banner("SUMMARY", char="=")
print(f"PDFs on disk:                {len(pdfs)}")
if oracle_ok:
    print(f"Papers in Oracle:            {paper_count}")
    print(f"Chunks in Oracle:            {chunk_count}")
    print(f"Chunks with native VECTOR:   {vec_count}")
    if chunk_count > 0:
        print(f"Embedding coverage:          {vec_count / chunk_count * 100:.1f}%")
if mem_path.exists():
    import sqlite3
    try:
        c = sqlite3.connect(str(mem_path))
        s = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        t = c.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        g = c.execute("SELECT COUNT(*) FROM facts WHERE scope='global'").fetchone()[0]
        c.close()
        print(f"Conversations in memory.db:  {s}")
        print(f"Total messages stored:       {t}")
        print(f"Long-term facts:             {g}")
    except Exception:
        pass
print()
print("This script is READ-ONLY. No data was modified.")
print("=" * 72)
