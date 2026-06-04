import os
from dotenv import load_dotenv
import oracledb

load_dotenv()

conn = oracledb.connect(
    user=os.getenv("ORACLE_USER"),
    password=os.getenv("ORACLE_PASSWORD"),
    dsn=os.getenv("ORACLE_DSN"),
)

cur = conn.cursor()

for table in ["PAPERS", "CHUNKS"]:
    print("\n" + "=" * 80)
    print(table)
    print("=" * 80)

    cur.execute("""
        SELECT column_name, data_type, data_length, nullable
        FROM user_tab_columns
        WHERE table_name = :table_name
        ORDER BY column_id
    """, {"table_name": table})

    for row in cur.fetchall():
        print(row)

cur.close()
conn.close()