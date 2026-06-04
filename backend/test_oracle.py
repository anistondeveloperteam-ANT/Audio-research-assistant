import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

print("Connecting to Oracle...")
print("DSN:", os.getenv("ORACLE_DSN"))

conn = oracledb.connect(
    user=os.getenv("ORACLE_USER"),
    password=os.getenv("ORACLE_PASSWORD"),
    dsn=os.getenv("ORACLE_DSN"),
)

cur = conn.cursor()
cur.execute("SELECT 'Oracle connection working' FROM dual")

print(cur.fetchone()[0])

cur.close()
conn.close()
print("Connection closed.")