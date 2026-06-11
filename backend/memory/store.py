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
import re
import sqlite3
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = 3


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

CREATE TABLE IF NOT EXISTS answer_cache (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               TEXT NOT NULL DEFAULT 'local',
    session_id            TEXT NOT NULL DEFAULT '',
    question              TEXT NOT NULL,
    normalized_question   TEXT NOT NULL,
    question_tokens_json  TEXT NOT NULL,
    answer                TEXT NOT NULL,
    sources_json          TEXT,
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL,
    last_used_at          REAL,
    hit_count             INTEGER NOT NULL DEFAULT 0,
    question_embedding    TEXT,
    embedding_meta        TEXT,
    UNIQUE (user_id, normalized_question)
);

CREATE INDEX IF NOT EXISTS idx_answer_cache_user_updated
    ON answer_cache(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_answer_cache_user_norm
    ON answer_cache(user_id, normalized_question);
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
    # Add semantic-cache columns to a pre-existing answer_cache table.
    ac_cols = {r[1] for r in cur.execute("PRAGMA table_info(answer_cache)").fetchall()}
    if ac_cols and "question_embedding" not in ac_cols:
        cur.execute("ALTER TABLE answer_cache ADD COLUMN question_embedding TEXT")
    if ac_cols and "embedding_meta" not in ac_cols:
        cur.execute("ALTER TABLE answer_cache ADD COLUMN embedding_meta TEXT")
    # Give upgraded DBs the same per-user dedup backstop as fresh ones: collapse any
    # pre-existing duplicate (user_id, normalized_question) rows, then add the index.
    if ac_cols:
        try:
            cur.execute("DELETE FROM answer_cache WHERE id NOT IN "
                        "(SELECT MAX(id) FROM answer_cache GROUP BY user_id, normalized_question)")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_answer_cache_unique_user_norm "
                        "ON answer_cache(user_id, normalized_question)")
        except Exception:
            pass
    current = cur.execute("PRAGMA user_version;").fetchone()[0]
    if current < SCHEMA_VERSION:
        cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    conn.commit()


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "describe", "does", "explain", "for", "from", "give", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "show", "tell",
    "that", "the", "this", "to", "use", "using", "what", "when", "where",
    "which", "why", "with", "you",
}


def normalize_question(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def question_tokens(text: str) -> List[str]:
    norm = normalize_question(text)
    out = []
    for tok in norm.split():
        if len(tok) <= 1 or tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ing"):
            tok = tok[:-3]
        elif len(tok) > 4 and tok.endswith("ed"):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


def question_similarity(a: str, b: str) -> float:
    """Cheap local similarity for cache reuse. No model/API call."""
    na = normalize_question(a)
    nb = normalize_question(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    seq = SequenceMatcher(None, na, nb).ratio()
    ta = question_tokens(na)
    tb = question_tokens(nb)
    if not ta or not tb:
        return seq

    ca = Counter(ta)
    cb = Counter(tb)
    shared = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in shared)
    na_len = sum(v * v for v in ca.values()) ** 0.5
    nb_len = sum(v * v for v in cb.values()) ** 0.5
    cosine = dot / max(na_len * nb_len, 1e-9)
    jaccard = len(set(ta) & set(tb)) / max(len(set(ta) | set(tb)), 1)

    # Require both word overlap and phrasing similarity to avoid overly broad
    # cache hits, but let exact-ish rephrases through.
    return max(seq * 0.45 + cosine * 0.45 + jaccard * 0.10, min(seq, cosine))


def unsafe_to_reuse(a: str, b: str) -> bool:
    """Block reuse of a *different* question even when the similarity score is high.

    A high lexical/semantic score does NOT mean the same question — e.g.
    "A vs B" / "B vs A" (a swap) or "A100" / "H100" (an identifier change) score
    near 1.0 yet have opposite/different answers. We err toward a cache MISS
    (regenerate) over serving the wrong answer. Returns True if reuse is unsafe.
    """
    sa, sb = set(question_tokens(a)), set(question_tokens(b))
    # Raw normalized tokens keep single-char entities (X, Y, A) and word order that
    # the stemmer drops — essential for catching swaps and 1-letter entity changes.
    ra, rb = normalize_question(a).split(), normalize_question(b).split()
    # Tokens containing a digit are identifiers (a100, gpt-4, ipv6) — must match.
    ids = lambda toks: {t for t in toks if any(c.isdigit() for c in t)}
    if ids(ra) != ids(rb) or ids(sa) != ids(sb):
        return True
    # Same tokens, different order = an argument swap ("a vs b" / "b vs a").
    if sorted(ra) == sorted(rb) and ra != rb:
        return True
    # A short, non-stopword distinguishing token differs (entity letters /
    # abbreviations like A/B, TCP/UDP, GPT/BERT) — likely a different subject.
    raw_diff = set(ra) ^ set(rb)
    if any(len(t) <= 3 and t not in _STOPWORDS for t in raw_diff):
        return True
    if any(len(t) <= 3 for t in (sa ^ sb)):
        return True
    # Polarity flip: one side negates/contrasts and the other doesn't — opposite
    # meaning despite a high similarity score ("with X" vs "without X",
    # "advantages" vs "disadvantages", "stable" vs "unstable").
    if (set(ra) & _NEG_WORDS) != (set(rb) & _NEG_WORDS):
        return True
    raw_all = set(ra) | set(rb)
    for t in raw_diff:
        for p in _NEG_PREFIXES:
            if t.startswith(p) and len(t) > len(p) + 3 and t[len(p):] in raw_all:
                return True
    # Exactly one content word swapped between otherwise-identical questions almost
    # always changes the answer (encoder/decoder, input/output, increase/decrease,
    # list/dict, gaming/mining, km/miles). Err toward a miss over a wrong answer.
    if len(sa - sb) == 1 and len(sb - sa) == 1:
        return True
    # Known antonym / contrast / unit groups referenced with DIFFERENT members.
    for grp in _CONTRAST_GROUPS:
        ga, gb = set(ra) & grp, set(rb) & grp
        if ga and gb and ga != gb:
            return True
    return False


_NEG_WORDS = {
    "not", "no", "without", "never", "except", "none", "nor", "cannot", "cant",
    "dont", "doesnt", "isnt", "arent", "wont", "vs", "versus",
}
_NEG_PREFIXES = ("dis", "un", "non", "anti", "ir", "im", "in")
_CONTRAST_GROUPS = [frozenset(_g.split()) for _g in (
    "increase decrease reduce raise lower amplify attenuate boost cut higher",
    "minimum maximum min max smallest largest highest lowest",
    "input output",
    "encode decode encoder decoder encrypt decrypt encryption decryption encoding decoding",
    "forward backward forwards backwards",
    "analog digital analogue",
    "lossless lossy",
    "before after",
    "append prepend",
    "symmetric asymmetric",
    "enable disable enabled disabled",
    "upsampling downsampling upsample downsample upsampled downsampled",
    "benefits drawbacks advantages disadvantages pros cons benefit drawback advantage disadvantage",
    "synchronous asynchronous sync async",
    "internal external",
    "compression decompression compress decompress",
    "km kilometer kilometers mile miles meter meters centimeter centimeters mm cm inch inches",
)]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity. embed_query() returns L2-normalized vectors, so this is a
    dot product; we still divide by norms for safety against unnormalized inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / max(na * nb, 1e-9)


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

    def __init__(self, db_path: Path, conversations_path: Optional[Path] = None):
        # Conversations (sessions/turns/facts) live in their own file so answer-cache
        # churn never contends with the chat history. The cache (answer_cache) stays in
        # the original db (db_path) and is ATTACHed as `cache` in every connection, so
        # the existing cross-table SQL keeps working in one transaction.
        self.cache_path = Path(db_path)
        self.conv_path = (Path(conversations_path) if conversations_path
                          else self.cache_path.parent / "conversations.db")
        self.db_path = self.conv_path
        self._split = self.conv_path != self.cache_path
        # cache.db keeps answer_cache (+ legacy conversation tables as an untouched backup).
        conn = _open_conn(self.cache_path)
        try:
            _migrate(conn)
        finally:
            conn.close()
        if self._split:
            # conversations.db holds sessions/turns/facts; it must NOT carry an answer_cache
            # table, so that an unqualified `answer_cache` resolves to the ATTACHed cache db.
            conn = _open_conn(self.conv_path)
            try:
                _migrate(conn)
                conn.execute("DROP TABLE IF EXISTS answer_cache")
                conn.commit()
            finally:
                conn.close()
        self._migrate_conversations()

    @contextmanager
    def _conn(self):
        """A connection with conversations.db as main and memory.db ATTACHed as `cache`
        (so unqualified `answer_cache` and cross-table SQL keep working in one transaction).

        Auto-commits on clean exit; rolls back and re-raises on error; always closes."""
        conn = _open_conn(self.conv_path)
        try:
            if self._split:
                conn.execute("ATTACH DATABASE ? AS cache", (str(self.cache_path),))
            conn.execute("PRAGMA foreign_keys=ON;")
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

    def _migrate_conversations(self) -> None:
        """One-time, idempotent copy of legacy conversation rows (sessions/turns/facts)
        from the old single db into conversations.db. Never deletes the source."""
        if self.conv_path == self.cache_path:
            return
        with self._conn() as conn:
            done = conn.execute(
                "SELECT 1 FROM facts WHERE scope = 'global' "
                "AND key = '_migrated_from_memory_db' LIMIT 1"
            ).fetchone()
            if done:
                return
            for table in ("sessions", "turns", "facts"):
                try:
                    conn.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM cache.{table}")
                except Exception:
                    pass
            now = time.time()
            conn.execute(
                "INSERT OR IGNORE INTO facts (scope, session_id, key, value, created_at, updated_at) "
                "VALUES ('global', NULL, '_migrated_from_memory_db', '1', ?, ?)", (now, now))

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

    def reassign_sessions(self, from_user: str, to_user: str) -> int:
        """Move every conversation owned by `from_user` to `to_user`; returns the
        count moved. Turns follow their session, so the whole chat moves with it.
        Used to fold pre-auth ('local') chats into a signed-in account so nothing
        disappears when login state changes."""
        if not from_user or not to_user or from_user == to_user:
            return 0
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE sessions SET user_id = ? WHERE user_id = ?",
                (to_user, from_user),
            )
            return cur.rowcount

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
                "DELETE FROM answer_cache WHERE session_id = ?",
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
            user_turn = conn.execute(
                "SELECT content FROM turns WHERE session_id = ? "
                "AND turn_index = ? AND role = 'user'",
                (session_id, turn_index),
            ).fetchone()
            nxt = conn.execute(
                "SELECT turn_index, role FROM turns "
                "WHERE session_id = ? AND turn_index > ? "
                "ORDER BY turn_index ASC LIMIT 1",
                (session_id, turn_index),
            ).fetchone()
            indices = [turn_index]
            if nxt and nxt["role"] == "assistant":
                indices.append(nxt["turn_index"])
            if user_turn:
                conn.execute(
                    "DELETE FROM answer_cache WHERE normalized_question = ? "
                    "AND user_id = (SELECT user_id FROM sessions WHERE id = ?)",
                    (normalize_question(user_turn["content"]), session_id),
                )
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
            user_rows = conn.execute(
                "SELECT content FROM turns WHERE session_id = ? "
                "AND turn_index >= ? AND role = 'user'",
                (session_id, turn_index),
            ).fetchall()
            for row in user_rows:
                conn.execute(
                    "DELETE FROM answer_cache WHERE normalized_question = ? "
                    "AND user_id = (SELECT user_id FROM sessions WHERE id = ?)",
                    (normalize_question(row["content"]), session_id),
                )
            cur = conn.execute(
                "DELETE FROM turns WHERE session_id = ? AND turn_index >= ?",
                (session_id, turn_index),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
            return cur.rowcount

    # ------- Answer cache --------------------------------------------
    def cache_answer(
        self,
        *,
        user_id: str,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        embedding: Optional[List[float]] = None,
        embedding_meta: Optional[str] = None,
    ) -> Optional[int]:
        norm = normalize_question(question)
        if not norm or not (answer or "").strip():
            return None
        now = time.time()
        tokens = question_tokens(question)
        user = user_id or "local"
        # Only persist a vector together with its provider/model meta (a vector with
        # no meta can't be safely compared later).
        emb = json.dumps(embedding) if (embedding and embedding_meta) else None
        with self._conn() as conn:
            # One row per (user, normalized question): replace any prior copy
            # (from this or any other session) so lookup and invalidation are
            # consistently per-user.
            conn.execute(
                "DELETE FROM answer_cache WHERE user_id = ? AND normalized_question = ?",
                (user, norm),
            )
            conn.execute(
                "INSERT INTO answer_cache (user_id, session_id, question, "
                "normalized_question, question_tokens_json, answer, sources_json, "
                "created_at, updated_at, last_used_at, hit_count, "
                "question_embedding, embedding_meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?)",
                (
                    user,
                    session_id or "",
                    question,
                    norm,
                    json.dumps(tokens),
                    answer,
                    json.dumps(sources) if sources else None,
                    now,
                    now,
                    emb,
                    embedding_meta if emb else None,
                ),
            )
            row = conn.execute(
                "SELECT id FROM answer_cache WHERE user_id = ? AND normalized_question = ?",
                (user, norm),
            ).fetchone()
            return int(row["id"]) if row else None

    def find_cached_answer(
        self,
        *,
        user_id: str,
        question: str,
        min_similarity: float = 0.97,
        query_embedding: Optional[List[float]] = None,
        query_meta: Optional[str] = None,
        min_semantic: float = 0.88,
        max_age_seconds: Optional[float] = None,
        limit: int = 200,
    ) -> Optional[Dict[str, Any]]:
        """Return the best safely-reusable cached answer for this user, or None.

        A candidate is reused only if it clears the lexical bar (`min_similarity`)
        OR the semantic bar (`min_semantic`, when an embedding is available AND was
        produced by the same provider/model as the cached vector) AND passes the
        `unsafe_to_reuse` guard that blocks swaps/identifier changes.
        """
        now = time.time()
        user = user_id or "local"
        cutoff = None if max_age_seconds is None else now - max_age_seconds
        params: List[Any] = [user]
        sql = (
            "SELECT id, session_id, question, answer, sources_json, updated_at, "
            "last_used_at, hit_count, question_embedding, embedding_meta "
            "FROM answer_cache WHERE user_id = ?"
        )
        if cutoff is not None:
            sql += " AND updated_at >= ?"
            params.append(cutoff)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        best: Optional[Dict[str, Any]] = None
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        for row in rows:
            if unsafe_to_reuse(question, row["question"]):
                continue
            lex = question_similarity(question, row["question"])
            sem = None
            if (query_embedding and row["question_embedding"]
                    and query_meta is not None and row["embedding_meta"] == query_meta):
                try:
                    sem = cosine_similarity(query_embedding, json.loads(row["question_embedding"]))
                except Exception:
                    sem = None
            if not (lex >= min_similarity or (sem is not None and sem >= min_semantic)):
                continue
            score = max(lex, sem or 0.0)
            if best is None or score > best["similarity"]:
                d = dict(row)
                d.pop("question_embedding", None)
                d.pop("embedding_meta", None)
                sj = d.pop("sources_json", None)
                d["sources"] = json.loads(sj) if sj else []
                d["similarity"] = float(score)
                d["match_kind"] = ("semantic" if (sem is not None and sem >= min_semantic
                                                  and sem >= lex) else "lexical")
                best = d
        return best

    def record_answer_cache_hit(self, cache_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE answer_cache SET hit_count = hit_count + 1, "
                "last_used_at = ? WHERE id = ?",
                (time.time(), int(cache_id)),
            )

    def clear_answer_cache(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        clauses = []
        params: List[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id or "local")
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id or "")
        sql = "DELETE FROM answer_cache"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._conn() as conn:
            cur = conn.execute(sql, tuple(params))
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
            a = conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
            g = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE scope = 'global'"
            ).fetchone()[0]
            s = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE scope = 'session'"
            ).fetchone()[0]
        return {
            "sessions": c,
            "turns": t,
            "answer_cache": a,
            "global_facts": g,
            "session_facts": s,
        }

    def close(self) -> None:
        # Nothing to close -- connections are per-call.
        # Keep the method for backward compatibility.
        pass
