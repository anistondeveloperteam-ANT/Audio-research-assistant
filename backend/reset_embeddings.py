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

try:
    cur.execute("UPDATE chunks SET embedding = NULL")
    conn.commit()
    print("All chunk embeddings cleared.")
except Exception as e:
    print("Failed to clear embeddings:", e)

cur.close()
conn.close()