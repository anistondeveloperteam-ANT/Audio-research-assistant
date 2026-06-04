import os
import sys
import oracledb
from dotenv import load_dotenv

load_dotenv()

if "--yes" not in sys.argv:
    print("This will delete indexed papers/chunks/concepts from Oracle.")
    print("It will NOT delete your PDF files.")
    print("Run with:")
    print("python backend\\reset_index.py --yes")
    raise SystemExit

conn = oracledb.connect(
    user=os.getenv("ORACLE_USER"),
    password=os.getenv("ORACLE_PASSWORD"),
    dsn=os.getenv("ORACLE_DSN"),
)

cur = conn.cursor()

for table in ["chunk_concepts", "concepts", "chunks", "papers"]:
    try:
        cur.execute(f"DELETE FROM {table}")
        print(f"Cleared {table}")
    except Exception as e:
        print(f"Could not clear {table}: {e}")

conn.commit()
cur.close()
conn.close()

print("Index reset complete. PDF files are untouched.")