"""
backend/cost_tracker.py  --  Track LLM API usage (Batch 12C)

Records every paid API call to data/llm_costs.db:
  - timestamp, provider, model, tokens_in, tokens_out, cost_usd

Provides:
  - record_call(...) -- log a call
  - get_today_cost() -- sum of today's spend in USD
  - get_session_cost(session_start) -- since a given timestamp
  - get_recent_calls(limit) -- last N calls for debugging

This module does NOT enforce hard spending limits (user opted out).
But it shows you what you've spent so you can self-regulate.

Storage: SQLite at data/llm_costs.db. Separate from memory.db so
crashes in one don't affect the other. WAL mode, per-call connection.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional


# Database lives in data/ alongside memory.db
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DATA_DIR / "llm_costs.db"


def _ensure_db():
    """Initialize the DB if missing. Idempotent."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp
                ON llm_calls(timestamp);
            PRAGMA journal_mode=WAL;
        """)
        conn.commit()
    finally:
        conn.close()


def record_call(
    provider: str,
    model: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record a single LLM call. Silently ignores errors so we never
    break the actual chat flow."""
    try:
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            conn.execute(
                "INSERT INTO llm_calls "
                "(timestamp, provider, model, tokens_in, tokens_out, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), provider, model, tokens_in, tokens_out, cost_usd),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # NEVER break the chat flow over cost tracking
        pass


def get_today_cost() -> float:
    """Return total USD spent since midnight (local time)."""
    try:
        # Compute midnight timestamp
        midnight = time.mktime(time.strptime(
            time.strftime("%Y-%m-%d"), "%Y-%m-%d"
        ))
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_calls "
                "WHERE timestamp >= ?",
                (midnight,)
            ).fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()
    except Exception:
        return 0.0


def get_session_cost(since_timestamp: float) -> float:
    """Return USD spent since the given timestamp."""
    try:
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_calls "
                "WHERE timestamp >= ?",
                (since_timestamp,)
            ).fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()
    except Exception:
        return 0.0


def get_recent_calls(limit: int = 20) -> List[Dict]:
    """Return the most recent N calls as a list of dicts."""
    try:
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT timestamp, provider, model, tokens_in, tokens_out, cost_usd "
                "FROM llm_calls ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        return []


def get_total_lifetime_cost() -> float:
    """Total USD ever spent through this app."""
    try:
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_calls"
            ).fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()
    except Exception:
        return 0.0


def format_usd(cents: float) -> str:
    """Format USD nicely. Sub-cent amounts shown as fractions."""
    if cents == 0:
        return "$0.00"
    if cents < 0.01:
        return f"${cents:.4f}"
    return f"${cents:.3f}"
