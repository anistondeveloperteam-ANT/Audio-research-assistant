import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

conn = oracledb.connect(
    user=os.getenv("ORACLE_USER"),
    password=os.getenv("ORACLE_PASSWORD"),
    dsn=os.getenv("ORACLE_DSN"),
)

cur = conn.cursor()

print("Database status\n")

cur.execute("SELECT COUNT(*) FROM papers")
print("Papers:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM chunks")
print("Chunks:", cur.fetchone()[0])

print("\nPapers:")
cur.execute("""
    SELECT id, title, page_count
    FROM papers
    ORDER BY id
""")

for row in cur.fetchall():
    print(f"{row[0]}. {row[1]} | pages={row[2]}")

print("\nTop chunk concept samples:")
cur.execute("""
    SELECT p.title, c.section_name, c.chunk_type, c.audio_concepts
    FROM chunks c
    JOIN papers p ON p.id = c.paper_id
    WHERE c.audio_concepts IS NOT NULL
    FETCH FIRST 10 ROWS ONLY
""")

for row in cur.fetchall():
    print("-" * 80)
    print("Paper:", row[0])
    print("Section:", row[1])
    print("Type:", row[2])
    print("Concepts:", row[3])

cur.close()
conn.close()