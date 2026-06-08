import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

SYSTEM_USER = "system"
SYSTEM_PASSWORD = os.getenv("ORACLE_PASSWORD")
DSN = os.getenv("ORACLE_DSN")

APP_USER = os.getenv("ORACLE_USER", "AUDIO_RAG")
# Read from .env — never hard-code credentials in source.
APP_PASSWORD = os.getenv("ORACLE_PASSWORD", "change_me")

print("Connecting as SYSTEM...")

conn = oracledb.connect(
    user=SYSTEM_USER,
    password=SYSTEM_PASSWORD,
    dsn=DSN,
)

cur = conn.cursor()

commands = [
    f"""
    BEGIN
        EXECUTE IMMEDIATE 'CREATE USER {APP_USER} IDENTIFIED BY {APP_PASSWORD}';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE = -01920 THEN
                NULL;
            ELSE
                RAISE;
            END IF;
    END;
    """,
    f"ALTER USER {APP_USER} QUOTA UNLIMITED ON USERS",
    f"GRANT CREATE SESSION TO {APP_USER}",
    f"GRANT CREATE TABLE TO {APP_USER}",
    f"GRANT CREATE VIEW TO {APP_USER}",
    f"GRANT CREATE SEQUENCE TO {APP_USER}",
    f"GRANT CREATE PROCEDURE TO {APP_USER}",
]

for cmd in commands:
    cur.execute(cmd)
    print("OK")

conn.commit()
cur.close()
conn.close()

print(f"User ready: {APP_USER}")