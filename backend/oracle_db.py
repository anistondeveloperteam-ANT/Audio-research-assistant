import oracledb
from config import ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN


def connect():
    return oracledb.connect(
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        dsn=ORACLE_DSN,
    )


def fetch_one(sql, params=None):
    conn = connect()
    cur = conn.cursor()
    cur.execute(sql, params or {})
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def fetch_all(sql, params=None):
    conn = connect()
    cur = conn.cursor()
    cur.execute(sql, params or {})
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def execute(sql, params=None, commit=True):
    conn = connect()
    cur = conn.cursor()
    cur.execute(sql, params or {})
    if commit:
        conn.commit()
    cur.close()
    conn.close()