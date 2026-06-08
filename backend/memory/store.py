"""
memory.py  --  Batch 9 (Phase 2)  --  thread-safe revision

Three-tier memory backed by SQLite.

THREAD SAFETY FIX (vs the initial Batch 9 release):
   The previous version held a single sqlite3.Connection across the
   life of the MemoryStore object. That's fine for single-threaded
   scripts but breaks under Streamlit's reactive model -- when the
   user clicks anything that triggers st.cache_resource.clear() +
   st.rerun(), the rerun lands on a different thread and SQLite
   refuses to reuse a connection from another thread.

   The fix below opens a fresh sqlite3.Connection per method call.
   SQLite is in-process and file-based; per-call connect/close is
   measured in microseconds and is the standard pattern for SQLite
   inside any threaded or async framework.

Schema and public API are unchanged. Existing data/memory.db files
continue to work without migration.

Tiers:
   Tier 1 -- short-term (turns table)
   Tier 2 -- working (facts, scope='session')
   Tier 3 -- long-term (facts, scope='global')

Plus a sessions table tying them together.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT 'New conversation',
    user_id       TEXT NOT NULL DEFAULT 'local',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_index    INTEGER NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    sources_json  TEXT,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns(session_id, turn_index);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scope         TEXT NOT NULL CHECK (scope IN ('session', 'global')),
    session_id    TEXT,
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    UNIQUE (scope, session_id, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_scope_session
    ON facts(scope, session_id);
"""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def default_db_path(project_root: Path) -> Path:
    return Path(project_root) / "data" / "memory.db"


def _open_conn(db_path: Path) -> sqlite3.Connection:
    """Open a fresh connection. Caller is responsible for closing it.

    check_same_thread=False is set as belt-and-braces, though we close
    each connection on the same thread that opened it. WAL mode keeps
    concurrent readers safe."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(_SCHEMA_SQL)
    # Add user_id to pre-existing sessions tables (per-user conversation isolation).
    cols = {r[1] for r in cur.execute("PRAGMA table_info(sessions)").fetchall()}
    if "user_id" not in cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'")
    current = cur.execute("PRAGMA user_version;").fetchone()[0]
    if current < SCHEMA_VERSION:
        cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    conn.commit()


# ----------------------------------------------------------------------
# Public class
# ----------------------------------------------------------------------

class MemoryStore:
    """Thin facade over SQLite. Per-call connections for thread safety.

    All public methods open a fresh sqlite3.Connection at start and
    close it at end. SQLite open/close is microseconds and is the
    correct pattern when the store is used from a framework that
    may invoke methods on different threads (Streamlit, FastAPI, etc.).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        # One-time migration. Opens its own connection.
        conn = _open_conn(self.db_path)
        try:
            _migrate(conn)
        finally:
            conn.close()

    @contextmanager
    def _conn(self):
        """Open a fresh connection for the duration of one operation.

        Auto-commits on clean exit; rolls back and re-raises on error;
        always closes the connection."""
        conn = _open_conn(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ------- Sessions ------------------------------------------------
    def create_session(self, title: str = "New conversation",
                       user_id: str = "local") -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, user_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, title, user_id or "local", now, now),
            )
        return sid

    def session_owner(self, session_id: str) -> Optional[str]:
        """Return the user_id that owns a session, or None if it doesn't exist."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return row["user_id"] if row else None

    def touch_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )

    def rename_session(self, session_id: str, title: str) -> None:
        title = (title or "").strip()
        if not title:
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title[:80], time.time(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM facts WHERE scope = 'session' AND session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )

    def list_sessions(self, limit: int = 50,
                     user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if user_id is None:
                cur = conn.execute(
                    "SELECT id, title, user_id, created_at, updated_at "
                    "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )
            else:
                cur = conn.execute(
                    "SELECT id, title, user_id, created_at, updated_at "
                    "FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                )
            return [dict(r) for r in cur.fetchall()]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ------- Turns ---------------------------------------------------
    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"role must be user/assistant/system, got {role!r}")
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE session_id = ?",
                (session_id,),
            )
            next_idx = cur.fetchone()[0]
            conn.execute(
                "INSERT INTO turns (session_id, turn_index, role, content, "
                "sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    next_idx,
                    role,
                    content,
                    json.dumps(sources) if sources else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return next_idx

    def get_turns(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT turn_index, role, content, sources_json, created_at "
            "FROM turns WHERE session_id = ? ORDER BY turn_index ASC"
        )
        params: Tuple = (session_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (session_id, limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            sj = d.pop("sources_json", None)
            d["sources"] = json.loads(sj) if sj else None
            out.append(d)
        return out

    def get_recent_turns(
        self,
        session_id: str,
        n_messages: int = 6,
    ) -> List[Dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? "
                "ORDER BY turn_index DESC LIMIT ?",
                (session_id, n_messages),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_turns(self, session_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM turns WHERE session_id = ?",
                (session_id,),
            )
            return cur.rowcount

    def delete_turn_pair(self, session_id: str, turn_index: int) -> int:
        """Delete the turn at `turn_index` and, if the next turn is an assistant
        reply, that one too -- i.e. remove a single question + its answer.

        Returns the number of rows deleted. Leaves a gap in turn_index, which is
        harmless: get_turns orders by turn_index and append_turn uses MAX+1.
        """
        with self._conn() as conn:
            nxt = conn.execute(
                "SELECT turn_index, role FROM turns "
                "WHERE session_id = ? AND turn_index > ? "
                "ORDER BY turn_index ASC LIMIT 1",
                (session_id, turn_index),
            ).fetchone()
            indices = [turn_index]
            if nxt and nxt["role"] == "assistant":
                indices.append(nxt["turn_index"])
            placeholders = ",".join("?" * len(indices))
            cur = conn.execute(
                f"DELETE FROM turns WHERE session_id = ? AND turn_index IN ({placeholders})",
                (session_id, *indices),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
            return cur.rowcount

    def delete_turns_from(self, session_id: str, turn_index: int) -> int:
        """Delete the turn at `turn_index` and every turn after it. Used when a
        user edits an earlier question: the conversation is truncated at that
        point and re-generated from there.

        Returns the number of rows deleted.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM turns WHERE session_id = ? AND turn_index >= ?",
                (session_id, turn_index),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
            return cur.rowcount

    # ------- Facts ---------------------------------------------------
    def upsert_fact(
        self,
        scope: str,
        key: str,
        value: str,
        session_id: Optional[str] = None,
    ) -> None:
        if scope not in ("session", "global"):
            raise ValueError(f"scope must be session or global, got {scope!r}")
        if scope == "session" and not session_id:
            raise ValueError("session-scoped facts require session_id")
        now = time.time()
        sid = session_id if scope == "session" else ""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO facts (scope, session_id, key, value, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope, session_id, key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at",
                (scope, sid, key, value, now, now),
            )

    def get_fact(
        self,
        scope: str,
        key: str,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        sid = session_id if scope == "session" else ""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM facts WHERE scope = ? AND session_id = ? AND key = ?",
                (scope, sid, key),
            ).fetchone()
        return row["value"] if row else None

    def list_facts(
        self,
        scope: str,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sid = session_id if scope == "session" else ""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value, created_at, updated_at "
                "FROM facts WHERE scope = ? AND session_id = ? "
                "ORDER BY updated_at DESC",
                (scope, sid),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(
        self,
        scope: str,
        key: str,
        session_id: Optional[str] = None,
    ) -> bool:
        sid = session_id if scope == "session" else ""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE scope = ? AND session_id = ? AND key = ?",
                (scope, sid, key),
            )
            return cur.rowcount > 0

    def clear_session_facts(self, session_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE scope = 'session' AND session_id = ?",
                (session_id,),
            )
            return cur.rowcount

    # ------- LLM-prompt block + stats --------------------------------
    def build_memory_block(
        self,
        session_id: str,
        max_facts: int = 12,
        max_chars_per_fact: int = 220,
    ) -> str:
        global_rows = self.list_facts("global")
        session_rows = self.list_facts("session", session_id) if session_id else []
        if not global_rows and not session_rows:
            return ""

        def trunc(s: str) -> str:
            s = str(s).strip()
            return s if len(s) <= max_chars_per_fact else s[: max_chars_per_fact - 3] + "..."

        lines: List[str] = []
        if global_rows:
            lines.append("Known facts about this user / project:")
            for r in global_rows[:max_facts]:
                lines.append(f"- {r['key']}: {trunc(r['value'])}")
        if session_rows:
            lines.append("Notes from this conversation:")
            for r in session_rows[:max_facts]:
                lines.append(f"- {r['key']}: {trunc(r['value'])}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        with self._conn() as conn:
            c = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            t = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            g = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE scope = 'global'"
            ).fetchone()[0]
            s = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE scope = 'session'"
            ).fetchone()[0]
        return {
            "sessions": c,
            "turns": t,
            "global_facts": g,
            "session_facts": s,
        }

    def close(self) -> None:
        # Nothing to close -- connections are per-call.
        # Keep the method for backward compatibility.
        pass
