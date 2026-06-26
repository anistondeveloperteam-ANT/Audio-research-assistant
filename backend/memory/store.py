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
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 6


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT 'New conversation',
    user_id       TEXT NOT NULL DEFAULT 'local',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    -- Compact-memory rolling summary of OLDER turns (what's sent to the LLM, never shown to the
    -- user). mem_summary_upto = how many older active-path turns are already folded in.
    mem_summary       TEXT,
    mem_summary_upto  INTEGER NOT NULL DEFAULT 0,
    mem_summary_at    REAL
);

CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_index        INTEGER NOT NULL,
    role              TEXT NOT NULL,
    content           TEXT NOT NULL,
    sources_json      TEXT,
    created_at        REAL NOT NULL,
    -- ChatGPT-style versioning. A user question is a "node" (node_id groups its
    -- versions); each assistant answer links to the specific question version it
    -- answers via parent_version_id. is_active marks the selected version in a group.
    node_id           TEXT,
    parent_version_id INTEGER,
    version_index     INTEGER NOT NULL DEFAULT 1,
    is_active         INTEGER NOT NULL DEFAULT 1,
    -- Turn kind: NULL for a normal chat turn, 'agent' for a coding-agent run — so a reloaded run is
    -- rendered + REGENERATED via the agent (not the chat path).
    kind              TEXT
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
    -- Origin-independent ANSWER QUALITY: 1 = judged high-quality/verified (safe to reuse),
    -- 0 = not verified / downgraded by dissatisfaction. Reuse admits ONLY verified=1.
    verified              INTEGER NOT NULL DEFAULT 0,
    -- The ANSWERING-LOGIC version that produced this entry. Reuse requires logic_version >= the
    -- current version, so a deploy of answering fixes invalidates older entries (they re-answer on
    -- next access). 0 = produced before this marker existed (treated as stale).
    logic_version         INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, normalized_question)
);

CREATE INDEX IF NOT EXISTS idx_answer_cache_user_updated
    ON answer_cache(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_answer_cache_user_norm
    ON answer_cache(user_id, normalized_question);

-- Code-agent result memory: every run's outcome (verified / partial / failed) is recorded for the
-- developer failure-pattern report AND for VERIFIED-ONLY reuse as a starting point. The agent never
-- edits its own source; this is accumulated evidence, not autonomous behaviour change.
CREATE TABLE IF NOT EXISTS agent_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              TEXT NOT NULL DEFAULT 'local',
    task                 TEXT NOT NULL,
    normalized_task      TEXT NOT NULL,
    task_tokens_json     TEXT NOT NULL DEFAULT '[]',
    requirements         TEXT,
    task_type            TEXT,
    code                 TEXT,
    output               TEXT,
    verification         TEXT NOT NULL DEFAULT 'failed',
    verified             INTEGER NOT NULL DEFAULT 0,
    tests_passed         INTEGER NOT NULL DEFAULT 0,
    tests_total          INTEGER NOT NULL DEFAULT 0,
    hidden_passed        INTEGER NOT NULL DEFAULT 0,
    hidden_total         INTEGER NOT NULL DEFAULT 0,
    attempts_taken       INTEGER NOT NULL DEFAULT 0,
    stop_reason          TEXT,
    cheat_reasons_json   TEXT,
    diagnosis            TEXT,
    gate_fail            TEXT,
    failing_checks_json  TEXT,
    created_at           REAL NOT NULL,
    updated_at           REAL NOT NULL,
    last_used_at         REAL,
    reuse_count          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_user_verified
    ON agent_runs(user_id, verified, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_runs_created
    ON agent_runs(created_at DESC);

-- Experience / lessons memory: the agent LEARNS from its own runs. Each row is a distilled lesson —
-- a Reflexion-style "what went wrong -> the fix" (kind='mistake'), or a user-PREFERRED exemplar from a
-- regeneration (kind='preference') — recalled on SIMILAR future questions and scored
-- relevance x recency x confidence. Reinforced when it helps produce a verified answer; pruned when
-- stale + low-confidence. One row per (user, normalized question, kind); a fresh capture upgrades the
-- prior. No model training — pure recall-before-answer, gated by EXPERIENCE_MEMORY.
CREATE TABLE IF NOT EXISTS lessons (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               TEXT NOT NULL DEFAULT 'local',
    kind                  TEXT NOT NULL,
    question              TEXT NOT NULL,
    normalized_question   TEXT NOT NULL,
    question_tokens_json  TEXT NOT NULL DEFAULT '[]',
    content               TEXT NOT NULL,
    source                TEXT,
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL,
    last_used_at          REAL,
    hit_count             INTEGER NOT NULL DEFAULT 0,
    confidence            REAL NOT NULL DEFAULT 1.0,
    question_embedding    TEXT,
    embedding_meta        TEXT,
    logic_version         INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, normalized_question, kind)
);

CREATE INDEX IF NOT EXISTS idx_lessons_user_created
    ON lessons(user_id, created_at DESC);

-- Acquired-knowledge / GROWN CORPUS (Phase 2): the agent UPDATES its own RAG from VERIFIED runs. Each
-- row is one external finding (web / paper / patent / repo / online-pdf passage) that was CITED in a
-- verified answer, embedded so it is retrievable on FUTURE questions WITHOUT re-fetching — the corpus
-- grows day by day from the agent's own verified research. Recalled by relevance x recency x confidence;
-- re-capturing the SAME finding (cited again in a later verified answer) UPSERTs and STRENGTHENS it
-- (confidence up, embedding kept). One row per (user, content_hash); pruned when stale + weak. Gated by
-- CORPUS_GROWTH. Captured in the BACKGROUND so it adds zero latency to the answer.
CREATE TABLE IF NOT EXISTS learned_sources (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT 'local',
    content_hash        TEXT NOT NULL,
    source_type         TEXT NOT NULL DEFAULT 'web',
    title               TEXT NOT NULL DEFAULT '',
    url                 TEXT NOT NULL DEFAULT '',
    snippet             TEXT NOT NULL DEFAULT '',
    text                TEXT NOT NULL,
    provider            TEXT,
    published           TEXT,
    question            TEXT,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    last_used_at        REAL,
    hit_count           INTEGER NOT NULL DEFAULT 0,
    confidence          REAL NOT NULL DEFAULT 1.0,
    embedding           TEXT,
    embedding_meta      TEXT,
    logic_version       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_learned_sources_user_created
    ON learned_sources(user_id, created_at DESC);

-- Self-tuning config (Phase 3): live, PERSISTED overrides for the pipeline's numeric thresholds, set
-- ONLY by the eval-gated tuner when a candidate value PROVABLY improves an offline metric. One row per
-- tunable name; absence = use the env/default (so an empty table = stock behaviour). Fully reversible
-- (delete a row to revert). Read through a zero-latency in-process cache, never per-request from disk.
CREATE TABLE IF NOT EXISTS tuned_config (
    name        TEXT PRIMARY KEY,
    value       REAL NOT NULL,
    source      TEXT,
    updated_at  REAL NOT NULL
);

-- Every tuning TRIAL (proposed or applied) for auditability: what was tried, the metric before/after,
-- and whether it was accepted. A complete, reversible record of how the config drifted over time.
CREATE TABLE IF NOT EXISTS tuning_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    old_value     REAL,
    new_value     REAL,
    metric_before REAL,
    metric_after  REAL,
    accepted      INTEGER NOT NULL DEFAULT 0,
    applied       INTEGER NOT NULL DEFAULT 0,
    note          TEXT,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tuning_events_created
    ON tuning_events(created_at DESC);
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
    prior_version = cur.execute("PRAGMA user_version;").fetchone()[0]
    cur.executescript(_SCHEMA_SQL)
    # Add user_id to pre-existing sessions tables (per-user conversation isolation).
    cols = {r[1] for r in cur.execute("PRAGMA table_info(sessions)").fetchall()}
    if "user_id" not in cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'")
    # Compact-memory rolling-summary columns (additive; safe on pre-existing sessions tables).
    if cols and "mem_summary" not in cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN mem_summary TEXT")
    if cols and "mem_summary_upto" not in cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN mem_summary_upto INTEGER NOT NULL DEFAULT 0")
    if cols and "mem_summary_at" not in cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN mem_summary_at REAL")
    # Add message-versioning columns to a pre-existing turns table, then backfill
    # legacy rows so every existing turn becomes version 1 (idempotent, non-destructive).
    turn_cols = {r[1] for r in cur.execute("PRAGMA table_info(turns)").fetchall()}
    if turn_cols and "node_id" not in turn_cols:
        cur.execute("ALTER TABLE turns ADD COLUMN node_id TEXT")
    if turn_cols and "parent_version_id" not in turn_cols:
        cur.execute("ALTER TABLE turns ADD COLUMN parent_version_id INTEGER")
    if turn_cols and "version_index" not in turn_cols:
        cur.execute("ALTER TABLE turns ADD COLUMN version_index INTEGER NOT NULL DEFAULT 1")
    if turn_cols and "is_active" not in turn_cols:
        cur.execute("ALTER TABLE turns ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
    if turn_cols and "kind" not in turn_cols:
        cur.execute("ALTER TABLE turns ADD COLUMN kind TEXT")
    # Answer-quality flag on the answer cache. Existing rows were cached under the old verified-only
    # gate, so backfill them to verified=1 (runs once, when the column is first added).
    ac_cols = {r[1] for r in cur.execute("PRAGMA table_info(answer_cache)").fetchall()}
    if ac_cols and "verified" not in ac_cols:
        cur.execute("ALTER TABLE answer_cache ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")
        cur.execute("UPDATE answer_cache SET verified = 1")
    # Answering-logic version marker. Existing rows default to 0 (= produced by older logic), so they
    # are treated as stale and re-answered on next access — that is how a deploy of answering fixes
    # takes effect instead of replaying outdated cached answers forever.
    if ac_cols and "logic_version" not in ac_cols:
        cur.execute("ALTER TABLE answer_cache ADD COLUMN logic_version INTEGER NOT NULL DEFAULT 0")
    # Created AFTER the columns exist (a legacy turns table gets them via ALTER above).
    if turn_cols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_turns_node ON turns(session_id, node_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_turns_parent ON turns(parent_version_id)")
    # Backfill ONLY when upgrading from a pre-versioning schema (< 4). After that, answers
    # legitimately carry a NULL node_id, so re-running would wrongly re-version them.
    if prior_version < 4:
        _backfill_turn_versions(cur)
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
    if prior_version < SCHEMA_VERSION:
        cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    conn.commit()


def _backfill_turn_versions(cur: sqlite3.Cursor) -> None:
    """Make legacy single-version turns into version 1 of a node. Runs once, when upgrading
    from a pre-versioning schema (gated on user_version in _migrate). Each user turn becomes
    its own question node; each assistant/system turn links to the most recent preceding user
    turn in the same session (reconstructing the old linear Q -> A pairing)."""
    try:
        rows = cur.execute(
            "SELECT id, session_id, role FROM turns WHERE node_id IS NULL "
            "ORDER BY session_id, turn_index, id"
        ).fetchall()
    except sqlite3.OperationalError:
        return  # turns table not present yet (e.g. cache db before any schema)
    last_user: Dict[str, int] = {}
    for r in rows:
        rid, sid, role = r["id"], r["session_id"], r["role"]
        if role == "user":
            node = uuid.uuid4().hex[:16]
            cur.execute(
                "UPDATE turns SET node_id = ?, parent_version_id = NULL, "
                "version_index = 1, is_active = 1 WHERE id = ?",
                (node, rid),
            )
            last_user[sid] = rid
        else:
            cur.execute(
                "UPDATE turns SET node_id = NULL, parent_version_id = ?, "
                "version_index = 1, is_active = 1 WHERE id = ?",
                (last_user.get(sid), rid),
            )


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


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token) for budgeting the assembled
    LLM context. Good enough to keep memory under a cap without a tokenizer dependency."""
    return (len(text or "") + 3) // 4


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
                conn.execute("DROP TABLE IF EXISTS agent_runs")
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
    @staticmethod
    def _new_node_id() -> str:
        return uuid.uuid4().hex[:16]

    def _insert_turn(self, conn, session_id, role, content, sources,
                     node_id, parent_version_id, version_index, is_active,
                     kind: Optional[str] = None) -> Dict[str, Any]:
        """Low-level versioned insert (caller owns the transaction). Returns id/turn_index."""
        now = time.time()
        next_idx = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        cur = conn.execute(
            "INSERT INTO turns (session_id, turn_index, role, content, sources_json, "
            "created_at, node_id, parent_version_id, version_index, is_active, kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, next_idx, role, content, json.dumps(sources) if sources else None,
             now, node_id, parent_version_id, version_index, int(is_active), kind),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return {"id": int(cur.lastrowid), "turn_index": int(next_idx), "version_index": version_index}

    @staticmethod
    def _active_user_version_id(conn, session_id: str) -> Optional[int]:
        """The id of the most recent ACTIVE user question version in the session."""
        row = conn.execute(
            "SELECT id FROM turns WHERE session_id = ? AND role = 'user' AND node_id IS NOT NULL "
            "AND is_active = 1 ORDER BY turn_index DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return int(row["id"]) if row else None

    @staticmethod
    def _next_answer_index(conn, qv_id: int) -> int:
        return int(conn.execute(
            "SELECT COALESCE(MAX(version_index), 0) + 1 FROM turns WHERE parent_version_id = ?",
            (qv_id,),
        ).fetchone()[0])

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Append a turn (back-compatible). A 'user' turn starts a fresh question node
        (version 1); an 'assistant'/'system' turn becomes a new answer version under the
        session's current active question, so the existing append-user-then-append-assistant
        flow keeps producing a clean Q -> A pair. Returns the turn_index (as before)."""
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"role must be user/assistant/system, got {role!r}")
        with self._conn() as conn:
            if role == "user":
                info = self._insert_turn(conn, session_id, role, content, None,
                                         node_id=self._new_node_id(), parent_version_id=None,
                                         version_index=1, is_active=1)
            else:
                qv = self._active_user_version_id(conn, session_id)
                if qv is None:                         # nothing to answer (e.g. a lone system turn)
                    info = self._insert_turn(conn, session_id, role, content, sources,
                                             node_id=None, parent_version_id=None,
                                             version_index=1, is_active=1)
                else:
                    conn.execute("UPDATE turns SET is_active = 0 WHERE parent_version_id = ?", (qv,))
                    info = self._insert_turn(conn, session_id, role, content, sources,
                                             node_id=None, parent_version_id=qv,
                                             version_index=self._next_answer_index(conn, qv),
                                             is_active=1)
        return info["turn_index"]

    # ------- Versioning (ChatGPT-style edit / regenerate / re-ask) ---
    def start_question(self, session_id: str, content: str) -> Dict[str, Any]:
        """Create a brand-new question node (version 1, active)."""
        with self._conn() as conn:
            node = self._new_node_id()
            info = self._insert_turn(conn, session_id, "user", content, None,
                                     node_id=node, parent_version_id=None,
                                     version_index=1, is_active=1)
        return {"node_id": node, "turn_id": info["id"], "version_index": 1, "total": 1}

    def add_question_version(self, session_id: str, node_id: str, content: str) -> Dict[str, Any]:
        """Add a new version of an existing question node (edit / re-ask). Becomes active;
        prior versions are kept but deactivated. Their answers are left untouched."""
        with self._conn() as conn:
            vi = int(conn.execute(
                "SELECT COALESCE(MAX(version_index), 0) + 1 FROM turns "
                "WHERE session_id = ? AND node_id = ?",
                (session_id, node_id),
            ).fetchone()[0])
            conn.execute("UPDATE turns SET is_active = 0 WHERE session_id = ? AND node_id = ?",
                         (session_id, node_id))
            info = self._insert_turn(conn, session_id, "user", content, None,
                                     node_id=node_id, parent_version_id=None,
                                     version_index=vi, is_active=1)
        return {"node_id": node_id, "turn_id": info["id"], "version_index": vi, "total": vi}

    def add_answer_version(self, question_version_id: int, content: str,
                           sources: Optional[List[Dict[str, Any]]] = None,
                           role: str = "assistant", kind: Optional[str] = None) -> Dict[str, Any]:
        """Add a new answer version under a specific question version (regenerate). Becomes
        active; prior answers are kept but deactivated. `kind='agent'` marks a coding-agent run."""
        with self._conn() as conn:
            row = conn.execute("SELECT session_id FROM turns WHERE id = ?",
                               (question_version_id,)).fetchone()
            if not row:
                raise ValueError(f"unknown question_version_id {question_version_id!r}")
            sid = row["session_id"]
            vi = self._next_answer_index(conn, question_version_id)
            conn.execute("UPDATE turns SET is_active = 0 WHERE parent_version_id = ?",
                         (question_version_id,))
            info = self._insert_turn(conn, sid, role, content, sources,
                                     node_id=None, parent_version_id=question_version_id,
                                     version_index=vi, is_active=1, kind=kind)
        return {"turn_id": info["id"], "version_index": vi, "total": vi}

    def set_active_question_version(self, session_id: str, node_id: str, version_index: int) -> bool:
        """Persist which question version the user is viewing (restored on reload)."""
        with self._conn() as conn:
            target = conn.execute(
                "SELECT id FROM turns WHERE session_id = ? AND node_id = ? AND version_index = ?",
                (session_id, node_id, version_index),
            ).fetchone()
            if not target:
                return False
            conn.execute("UPDATE turns SET is_active = 0 WHERE session_id = ? AND node_id = ?",
                         (session_id, node_id))
            conn.execute("UPDATE turns SET is_active = 1 WHERE id = ?", (target["id"],))
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?",
                         (time.time(), session_id))
        return True

    def set_active_answer_version(self, question_version_id: int, version_index: int) -> bool:
        """Persist which answer version (under a question version) the user is viewing."""
        with self._conn() as conn:
            target = conn.execute(
                "SELECT id, session_id FROM turns WHERE parent_version_id = ? AND version_index = ?",
                (question_version_id, version_index),
            ).fetchone()
            if not target:
                return False
            conn.execute("UPDATE turns SET is_active = 0 WHERE parent_version_id = ?",
                         (question_version_id,))
            conn.execute("UPDATE turns SET is_active = 1 WHERE id = ?", (target["id"],))
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?",
                         (time.time(), target["session_id"]))
        return True

    def get_version(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one version's content + sources (lazy-load when switching in the UI)."""
        with self._conn() as conn:
            r = conn.execute(
                "SELECT id, session_id, role, content, sources_json, node_id, "
                "parent_version_id, version_index, is_active, kind, created_at "
                "FROM turns WHERE id = ?",
                (turn_id,),
            ).fetchone()
        if not r:
            return None
        d = dict(r)
        sj = d.pop("sources_json", None)
        d["sources"] = json.loads(sj) if sj else None
        return d

    def get_conversation_tree(self, session_id: str) -> List[Dict[str, Any]]:
        """The full version tree for rendering: ordered question slots, each with all its
        question versions and (per version) all answer versions. Content is included only
        for the ACTIVE path; inactive versions carry refs (turn_id/version_index) so the UI
        lazy-loads them via get_version when switched to."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, turn_index, role, content, sources_json, node_id, "
                "parent_version_id, version_index, is_active, kind "
                "FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()

        answers_by_parent: Dict[int, List[Any]] = {}
        node_versions: Dict[str, List[Any]] = {}
        node_pos: Dict[str, int] = {}
        for r in rows:
            if r["parent_version_id"] is not None:
                answers_by_parent.setdefault(int(r["parent_version_id"]), []).append(r)
            if r["role"] == "user" and r["node_id"]:
                node_versions.setdefault(r["node_id"], []).append(r)
                node_pos[r["node_id"]] = min(node_pos.get(r["node_id"], r["turn_index"]),
                                             r["turn_index"])

        def _answer_entry(a, include_content: bool) -> Dict[str, Any]:
            sj = a["sources_json"]
            return {
                "turn_id": int(a["id"]),
                "version_index": int(a["version_index"]),
                "is_active": bool(a["is_active"]),
                "kind": a["kind"],
                "content": a["content"] if include_content else None,
                "sources": (json.loads(sj) if sj else None) if include_content else None,
            }

        slots: List[Dict[str, Any]] = []
        for node_id in sorted(node_pos, key=lambda n: node_pos[n]):
            qvs = sorted(node_versions[node_id], key=lambda r: r["version_index"])
            active_q = int(qvs[-1]["version_index"])
            for qv in qvs:
                if qv["is_active"]:
                    active_q = int(qv["version_index"])
            versions = []
            for qv in qvs:
                q_active = bool(qv["is_active"])
                ans = sorted(answers_by_parent.get(int(qv["id"]), []),
                             key=lambda r: r["version_index"])
                active_a = int(ans[-1]["version_index"]) if ans else 0
                for a in ans:
                    if a["is_active"]:
                        active_a = int(a["version_index"])
                versions.append({
                    "turn_id": int(qv["id"]),
                    "version_index": int(qv["version_index"]),
                    "is_active": q_active,
                    "content": qv["content"] if q_active else None,
                    "answer_total": len(ans),
                    "active_answer_index": active_a,
                    "answers": [_answer_entry(a, q_active and bool(a["is_active"])) for a in ans],
                })
            slots.append({
                "node_id": node_id,
                "version_total": len(qvs),
                "active_version_index": active_q,
                "versions": versions,
            })
        return slots

    def delete_node(self, session_id: str, node_id: str) -> int:
        """Delete an entire question node — all its versions and all their answers."""
        with self._conn() as conn:
            qv_ids = [int(r["id"]) for r in conn.execute(
                "SELECT id FROM turns WHERE session_id = ? AND node_id = ?",
                (session_id, node_id),
            ).fetchall()]
            deleted = 0
            if qv_ids:
                placeholders = ",".join("?" * len(qv_ids))
                cur = conn.execute(
                    f"DELETE FROM turns WHERE parent_version_id IN ({placeholders})", qv_ids)
                deleted += cur.rowcount
                cur = conn.execute(
                    "DELETE FROM turns WHERE session_id = ? AND node_id = ?",
                    (session_id, node_id))
                deleted += cur.rowcount
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?",
                         (time.time(), session_id))
            return deleted

    def get_turns(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """The ACTIVE linear conversation (currently-selected versions only), ordered by
        slot position then question/answer. Old single-version chats return exactly as before
        (every row is active version 1). Shape is unchanged: turn_index/role/content/sources."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, turn_index, role, content, sources_json, node_id, "
                "parent_version_id, is_active "
                "FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()

        node_pos: Dict[str, int] = {}
        active_q: Dict[str, Any] = {}
        active_ans: Dict[int, Any] = {}
        orphans: List[Any] = []
        for r in rows:
            if r["role"] == "user" and r["node_id"]:
                node_pos[r["node_id"]] = min(node_pos.get(r["node_id"], r["turn_index"]),
                                             r["turn_index"])
                if r["is_active"]:
                    active_q[r["node_id"]] = r
            elif r["parent_version_id"] is not None:
                if r["is_active"]:
                    active_ans[int(r["parent_version_id"])] = r
            elif r["is_active"]:
                orphans.append(r)               # standalone assistant/system (no question)

        # Order each slot by its node's ORIGINAL position (min turn_index), so editing an
        # earlier question keeps it in place; question before answer within a slot.
        entries: List[tuple] = []
        for node_id, qv in active_q.items():
            pos = node_pos[node_id]
            entries.append((pos, 0, qv))
            ans = active_ans.get(int(qv["id"]))
            if ans is not None:
                entries.append((pos, 1, ans))
        for r in orphans:
            entries.append((int(r["turn_index"]), 0, r))
        entries.sort(key=lambda e: (e[0], e[1]))

        def _row(r) -> Dict[str, Any]:
            sj = r["sources_json"]
            return {"turn_index": int(r["turn_index"]), "role": r["role"],
                    "content": r["content"], "sources": json.loads(sj) if sj else None}

        out = [_row(e[2]) for e in entries]
        if limit is not None:
            out = out[:limit]
        return out

    def get_recent_turns(
        self,
        session_id: str,
        n_messages: int = 6,
    ) -> List[Dict[str, str]]:
        """Recent turns along the ACTIVE path (so the LLM sees the currently-selected
        conversation, not stale/alternate versions)."""
        turns = self.get_turns(session_id)
        recent = turns[-n_messages:] if n_messages else turns
        return [{"role": t["role"], "content": t["content"]} for t in recent]

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
        verified: bool = True,
        logic_version: int = 0,
    ) -> Optional[int]:
        """Persist an answer for reuse, WITH its origin-independent quality status. `verified=True`
        (judged high-quality) makes it reusable; an unverified/low-quality answer is recorded but
        NEVER reused (find_cached_answer admits only verified=1). `logic_version` stamps the
        answering-logic version that produced it, so a later deploy can invalidate older entries.
        One row per (user, normalized question): a fresh verified answer UPGRADES (replaces) any
        prior low-quality record."""
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
                "question_embedding, embedding_meta, verified, logic_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?)",
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
                    1 if verified else 0,
                    int(logic_version),
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
        min_logic_version: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Return the best safely-reusable cached answer for this user, or None.

        A candidate is reused only if it clears the lexical bar (`min_similarity`)
        OR the semantic bar (`min_semantic`, when an embedding is available AND was
        produced by the same provider/model as the cached vector) AND passes the
        `unsafe_to_reuse` guard that blocks swaps/identifier changes. `min_logic_version`
        excludes entries produced by older answering logic, so a deploy of answering fixes
        forces those questions to re-answer instead of replaying a stale cached answer.
        """
        now = time.time()
        user = user_id or "local"
        cutoff = None if max_age_seconds is None else now - max_age_seconds
        params: List[Any] = [user]
        # ONLY verified=1 rows are reusable — a stored low-quality / dissatisfaction-downgraded answer
        # is never replayed; the caller re-answers fresh and may upgrade the record.
        sql = (
            "SELECT id, session_id, question, answer, sources_json, updated_at, "
            "last_used_at, hit_count, question_embedding, embedding_meta, logic_version "
            "FROM answer_cache WHERE user_id = ? AND verified = 1"
        )
        if min_logic_version > 0:
            sql += " AND logic_version >= ?"
            params.append(int(min_logic_version))
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

    def downgrade_cached_answer(self, user_id: str, question: str) -> bool:
        """Mark this question's cached answer NOT reusable (user dissatisfaction / regenerate), so the
        next matching query re-answers fresh instead of replaying it. A later high-quality answer
        re-upgrades the record via cache_answer. Returns True if a row was downgraded."""
        norm = normalize_question(question)
        if not norm:
            return False
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE answer_cache SET verified = 0 WHERE user_id = ? AND normalized_question = ?",
                (user_id or "local", norm),
            )
            return cur.rowcount > 0

    def mark_cache_unverified(self, cache_id: int) -> bool:
        """Mark a specific cached row NOT reusable by id (used when a serve-time re-check finds the
        stored answer inconsistent). By-id is required because a semantic match's stored question can
        differ from the asked one, so a normalized-question update would miss it. Returns True if a
        row was changed."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE answer_cache SET verified = 0 WHERE id = ?", (int(cache_id),))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # EXPERIENCE / LESSONS memory (the "learns day by day" layer). A lesson distilled from a run is
    # recalled on SIMILAR future questions (scored relevance x recency x confidence), injected into the
    # draft, reinforced when it yields a verified answer, and pruned when stale + weak. No model
    # training — pure recall-before-answer. Gated by EXPERIENCE_MEMORY at the caller.
    # ------------------------------------------------------------------
    def record_lesson(
        self,
        *,
        user_id: str,
        kind: str,
        question: str,
        content: str,
        source: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        embedding_meta: Optional[str] = None,
        confidence: float = 1.0,
        logic_version: int = 0,
    ) -> Optional[int]:
        """Store (or upgrade) one lesson. ONE row per (user, normalized question, kind): a fresh
        capture replaces the prior so the same question never accumulates duplicates. Returns the row
        id, or None when the question / content / kind is empty."""
        norm = normalize_question(question)
        text = (content or "").strip()
        if not norm or not text or not (kind or "").strip():
            return None
        now = time.time()
        user = user_id or "local"
        emb = json.dumps(embedding) if (embedding and embedding_meta) else None
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM lessons WHERE user_id = ? AND normalized_question = ? AND kind = ?",
                (user, norm, kind),
            )
            conn.execute(
                "INSERT INTO lessons (user_id, kind, question, normalized_question, "
                "question_tokens_json, content, source, created_at, updated_at, last_used_at, "
                "hit_count, confidence, question_embedding, embedding_meta, logic_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?)",
                (user, kind, question, norm, json.dumps(question_tokens(question)), text, source,
                 now, now, max(0.0, float(confidence)), emb, embedding_meta if emb else None,
                 int(logic_version)),
            )
            row = conn.execute(
                "SELECT id FROM lessons WHERE user_id = ? AND normalized_question = ? AND kind = ?",
                (user, norm, kind),
            ).fetchone()
        self.prune_lessons(user_id=user)                 # keep the table bounded (separate connection)
        return int(row["id"]) if row else None

    def recall_lessons(
        self,
        *,
        user_id: str,
        question: str,
        query_embedding: Optional[List[float]] = None,
        query_meta: Optional[str] = None,
        min_relevance: float = 0.62,
        top_k: int = 3,
        half_life_days: float = 30.0,
        scan_limit: int = 400,
    ) -> List[Dict[str, Any]]:
        """The most useful lessons for THIS question, most-useful first. Each candidate is scored
        relevance x recency x confidence: relevance = max(lexical, semantic-when-a-comparable-embedding-
        exists); recency is a half-life decay on age; confidence rises with reinforcement.

        Unlike the answer cache, recall does NOT apply unsafe_to_reuse: a lesson is GENERALISABLE
        guidance, not a stored answer, so it SHOULD carry across similar questions even when numbers /
        identifiers differ (a "compute it in code" lesson applies to 3-min and 5-min audio alike). The
        relevance floor is set ABOVE the word-overlap/different-intent band (~0.6) so a lesson can't be
        pulled into an unrelated-intent question that merely shares words — while a same-question/
        different-number paraphrase (~0.85+) still generalises. That floor, plus the fact that lessons
        are process/style guidance that can never inject a wrong fact (preference lessons are shape-only),
        is what keeps recall safe."""
        user = user_id or "local"
        now = time.time()
        hl = max(1.0, float(half_life_days))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, kind, question, content, source, created_at, last_used_at, hit_count, "
                "confidence, question_embedding, embedding_meta FROM lessons "
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user, max(1, int(scan_limit))),
            ).fetchall()
        scored: List[Dict[str, Any]] = []
        for row in rows:
            lex = question_similarity(question, row["question"])
            sem = None
            if (query_embedding and row["question_embedding"]
                    and query_meta is not None and row["embedding_meta"] == query_meta):
                try:
                    sem = cosine_similarity(query_embedding, json.loads(row["question_embedding"]))
                except Exception:
                    sem = None
            rel = max(lex, sem or 0.0)
            if rel < min_relevance:
                continue
            created = row["created_at"]
            age_days = max(0.0, (now - float(created if created is not None else now)) / 86400.0)
            recency = 0.5 ** (age_days / hl)
            # `or 1.0` would resurrect a decayed-to-0.0 lesson at full strength (0.0 is falsy); guard None only.
            conf_raw = row["confidence"]
            conf = max(0.0, float(conf_raw if conf_raw is not None else 1.0))
            d = {k: row[k] for k in ("id", "kind", "question", "content", "source",
                                     "created_at", "hit_count", "confidence")}
            d["relevance"] = float(rel)
            d["score"] = float(rel * recency * conf)
            scored.append(d)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(0, int(top_k))]

    def reinforce_lessons(self, lesson_ids: List[int], *, delta: float = 0.5, cap: float = 5.0) -> None:
        """A lesson that helped produce a VERIFIED answer gets stronger (confidence += delta, capped)
        and its use recorded — so proven lessons rise while unused ones fade via recency + prune."""
        ids = [int(i) for i in (lesson_ids or []) if i is not None]
        if not ids:
            return
        now = time.time()
        with self._conn() as conn:
            for lid in ids:
                conn.execute(
                    "UPDATE lessons SET confidence = MIN(?, confidence + ?), hit_count = hit_count + 1, "
                    "last_used_at = ?, updated_at = ? WHERE id = ?",
                    (float(cap), float(delta), now, now, lid),
                )

    def prune_lessons(self, *, user_id: str, max_per_user: int = 500) -> int:
        """Keep the table bounded: drop this user's lessons beyond `max_per_user`, evicting the WEAKEST
        first (lowest confidence, then oldest). Returns the count deleted."""
        user = user_id or "local"
        with self._conn() as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM lessons WHERE user_id = ?",
                             (user,)).fetchone()["c"]
            if n <= max_per_user:
                return 0
            conn.execute(
                "DELETE FROM lessons WHERE id IN ("
                "SELECT id FROM lessons WHERE user_id = ? ORDER BY confidence ASC, created_at ASC "
                "LIMIT ?)",
                (user, int(n - max_per_user)),
            )
            return int(n - max_per_user)

    # ------------------------------------------------------------------
    # Acquired-knowledge / GROWN CORPUS (Phase 2): store + recall external findings that a VERIFIED
    # answer CITED, so future questions retrieve them locally without re-fetching. Same embedding
    # pattern as lessons/answer_cache (caller passes the vector + provider/model meta tag).
    # ------------------------------------------------------------------
    def record_learned_source(
        self,
        *,
        user_id: str,
        content_hash: str,
        text: str,
        source_type: str = "web",
        title: str = "",
        url: str = "",
        snippet: str = "",
        provider: Optional[str] = None,
        published: Optional[str] = None,
        question: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        embedding_meta: Optional[str] = None,
        confidence: float = 1.0,
        logic_version: int = 0,
    ) -> Optional[int]:
        """Store (or STRENGTHEN) one acquired finding. ONE row per (user, content_hash): re-capturing the
        same finding (cited again in a later verified answer) UPSERTs — confidence rises, hit_count++,
        text/snippet refresh, and the existing embedding is KEPT when none is supplied (so we never
        re-embed needlessly). Returns the row id, or None when content_hash / text is empty."""
        ch = (content_hash or "").strip()
        body = (text or "").strip()
        if not ch or not body:
            return None
        now = time.time()
        user = user_id or "local"
        emb = json.dumps(embedding) if (embedding and embedding_meta) else None
        meta = embedding_meta if emb else None
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO learned_sources (user_id, content_hash, source_type, title, url, snippet, "
                "text, provider, published, question, created_at, updated_at, last_used_at, hit_count, "
                "confidence, embedding, embedding_meta, logic_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, content_hash) DO UPDATE SET "
                "  hit_count = hit_count + 1, "
                "  confidence = MIN(5.0, confidence + 0.5), "
                "  text = excluded.text, snippet = excluded.snippet, title = excluded.title, "
                "  url = excluded.url, source_type = excluded.source_type, "
                "  provider = COALESCE(excluded.provider, provider), "
                "  published = COALESCE(excluded.published, published), "
                "  question = COALESCE(excluded.question, question), "
                "  embedding = COALESCE(excluded.embedding, embedding), "
                "  embedding_meta = COALESCE(excluded.embedding_meta, embedding_meta), "
                "  logic_version = excluded.logic_version, "
                "  updated_at = excluded.updated_at, last_used_at = excluded.updated_at",
                (user, ch, (source_type or "web"), (title or ""), (url or ""), (snippet or "")[:600],
                 body, provider, published, question, now, now, max(0.0, float(confidence)), emb, meta,
                 int(logic_version)),
            )
            row = conn.execute(
                "SELECT id FROM learned_sources WHERE user_id = ? AND content_hash = ?",
                (user, ch),
            ).fetchone()
        return int(row["id"]) if row else None

    def existing_source_hashes(self, *, user_id: str, hashes: List[str]) -> set:
        """The subset of `hashes` already stored for this user — so capture skips re-embedding findings
        it has seen before (the embedding call is the only expensive step)."""
        hs = [h for h in (hashes or []) if h]
        if not hs:
            return set()
        user = user_id or "local"
        placeholders = ",".join("?" for _ in hs)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT content_hash FROM learned_sources WHERE user_id = ? AND content_hash IN ({placeholders})",
                (user, *hs),
            ).fetchall()
        return {r["content_hash"] for r in rows}

    def recall_learned_sources(
        self,
        *,
        user_id: str,
        question: str,
        query_embedding: Optional[List[float]] = None,
        query_meta: Optional[str] = None,
        min_relevance: float = 0.5,
        top_k: int = 3,
        half_life_days: float = 120.0,
        scan_limit: int = 600,
    ) -> List[Dict[str, Any]]:
        """The acquired passages most relevant to THIS question, most-useful first. Each is scored
        relevance x recency x confidence: relevance = max(lexical(question, title+snippet), semantic
        (query vs the passage's document embedding, when a comparable embedding exists)); recency is a
        half-life decay on how recently we learned/used it; confidence rises each time the passage proves
        useful. Semantic matching only fires when query_meta == the stored embedding_meta (same embedding
        source), exactly like the answer cache — otherwise it falls back to lexical."""
        user = user_id or "local"
        now = time.time()
        hl = max(1.0, float(half_life_days))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, content_hash, source_type, title, url, snippet, text, provider, published, "
                "question, created_at, hit_count, confidence, embedding, embedding_meta "
                "FROM learned_sources WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user, max(1, int(scan_limit))),
            ).fetchall()
        scored: List[Dict[str, Any]] = []
        for row in rows:
            # Score against the BODY we actually inject (snippet is a body excerpt), NOT the title — a
            # title that matches the question while the body is off-topic must NOT pull the passage in.
            # (Semantic already uses the body's own embedding.)
            lex_target = ((row["snippet"] or row["text"] or "")[:400]).strip()
            lex = question_similarity(question, lex_target)
            sem = None
            if (query_embedding and row["embedding"]
                    and query_meta is not None and row["embedding_meta"] == query_meta):
                try:
                    sem = cosine_similarity(query_embedding, json.loads(row["embedding"]))
                except Exception:
                    sem = None
            rel = max(lex, sem or 0.0)
            if rel < min_relevance:
                continue
            created = row["created_at"]
            age_days = max(0.0, (now - float(created if created is not None else now)) / 86400.0)
            recency = 0.5 ** (age_days / hl)
            conf_raw = row["confidence"]
            conf = max(0.0, float(conf_raw if conf_raw is not None else 1.0))
            d = {k: row[k] for k in ("id", "content_hash", "source_type", "title", "url", "snippet",
                                     "text", "provider", "published", "question", "created_at",
                                     "hit_count", "confidence")}
            d["relevance"] = float(rel)
            d["score"] = float(rel * recency * conf)
            scored.append(d)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(0, int(top_k))]

    def reinforce_learned_sources(self, source_ids: List[int], *, delta: float = 0.5,
                                  cap: float = 5.0) -> None:
        """Strengthen acquired passages that helped produce a verified answer (confidence += delta,
        capped; hit_count++). Re-capture also strengthens via UPSERT; this is the direct path."""
        ids = [int(i) for i in (source_ids or []) if i is not None]
        if not ids:
            return
        now = time.time()
        with self._conn() as conn:
            for sid in ids:
                conn.execute(
                    "UPDATE learned_sources SET confidence = MIN(?, confidence + ?), "
                    "hit_count = hit_count + 1, last_used_at = ?, updated_at = ? WHERE id = ?",
                    (float(cap), float(delta), now, now, sid),
                )

    def prune_learned_sources(self, *, user_id: str, max_per_user: int = 1000) -> int:
        """Keep the grown corpus bounded: drop this user's acquired passages beyond `max_per_user`,
        evicting the WEAKEST first (lowest confidence, then oldest). Returns the count deleted."""
        user = user_id or "local"
        with self._conn() as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM learned_sources WHERE user_id = ?",
                             (user,)).fetchone()["c"]
            if n <= max_per_user:
                return 0
            conn.execute(
                "DELETE FROM learned_sources WHERE id IN ("
                "SELECT id FROM learned_sources WHERE user_id = ? ORDER BY confidence ASC, created_at ASC "
                "LIMIT ?)",
                (user, int(n - max_per_user)),
            )
            return int(n - max_per_user)

    # ------------------------------------------------------------------
    # Self-tuning config (Phase 3): persisted numeric-threshold overrides + an audit trail of every
    # tuning trial. Only the eval-gated tuner writes these; an empty table means stock behaviour.
    # ------------------------------------------------------------------
    def get_tuned_config(self) -> Dict[str, float]:
        """All active threshold overrides as {name: value}. Empty dict = no overrides (stock defaults).
        Read by the tuning cache, not per-request."""
        with self._conn() as conn:
            rows = conn.execute("SELECT name, value FROM tuned_config").fetchall()
        return {r["name"]: float(r["value"]) for r in rows}

    def set_tuned_config(self, name: str, value: float, *, source: str = "self_tuner") -> None:
        """Persist (or update) one override. The caller (the tuner) is responsible for clamping to the
        tunable's bounds and for only doing this on an eval-proven improvement."""
        nm = (name or "").strip()
        if not nm:
            return
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tuned_config (name, value, source, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = excluded.value, source = excluded.source, "
                "updated_at = excluded.updated_at",
                (nm, float(value), source, now),
            )

    def clear_tuned_config(self, name: Optional[str] = None) -> int:
        """Revert one override (by name) or ALL of them (name=None) back to env/default. Returns the
        number of overrides removed. This is the one-call rollback for self-tuning."""
        with self._conn() as conn:
            if name is None:
                cur = conn.execute("DELETE FROM tuned_config")
            else:
                cur = conn.execute("DELETE FROM tuned_config WHERE name = ?", ((name or "").strip(),))
            return int(cur.rowcount if cur.rowcount is not None else 0)

    def record_tuning_event(self, *, name: str, old_value: Optional[float], new_value: Optional[float],
                            metric_before: Optional[float], metric_after: Optional[float],
                            accepted: bool, applied: bool = False, note: Optional[str] = None) -> None:
        """Audit one tuning trial (proposed or applied). Never raises on a None numeric."""
        def _f(x):
            return None if x is None else float(x)
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tuning_events (name, old_value, new_value, metric_before, metric_after, "
                "accepted, applied, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, _f(old_value), _f(new_value), _f(metric_before), _f(metric_after),
                 1 if accepted else 0, 1 if applied else 0, note, now),
            )

    def get_tuning_history(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """The most recent tuning trials, newest first — for the audit/UI."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, old_value, new_value, metric_before, metric_after, accepted, applied, "
                "note, created_at FROM tuning_events ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Code-agent result memory (learning layer). Every run is recorded for the developer
    # failure-pattern report; only VERIFIED runs are ever reused, and reuse only SEEDS a fresh
    # attempt that still passes the full gate stack. The agent never edits its own source.
    # ------------------------------------------------------------------
    def record_agent_run(
        self,
        *,
        user_id: str,
        task: str,
        code: str = "",
        output: str = "",
        verification: str = "failed",
        requirements: str = "",
        task_type: str = "",
        tests_passed: int = 0,
        tests_total: int = 0,
        hidden_passed: int = 0,
        hidden_total: int = 0,
        attempts_taken: int = 0,
        stop_reason: str = "",
        cheat_reasons: Optional[List[str]] = None,
        diagnosis: str = "",
        gate_fail: str = "",
        failing_checks: Optional[List[str]] = None,
    ) -> Optional[int]:
        """Persist one code-agent run's outcome. Returns the row id, or None on an empty task."""
        norm = normalize_question(task)
        if not norm:
            return None
        now = time.time()
        user = user_id or "local"
        verified = 1 if (verification or "") == "verified" else 0
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_runs (user_id, task, normalized_task, task_tokens_json, "
                "requirements, task_type, code, output, verification, verified, tests_passed, "
                "tests_total, hidden_passed, hidden_total, attempts_taken, stop_reason, "
                "cheat_reasons_json, diagnosis, gate_fail, failing_checks_json, created_at, "
                "updated_at, last_used_at, reuse_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)",
                (
                    user, task, norm, json.dumps(question_tokens(task)),
                    requirements or "", task_type or "", code or "", output or "",
                    verification or "failed", verified, int(tests_passed), int(tests_total),
                    int(hidden_passed), int(hidden_total), int(attempts_taken), stop_reason or "",
                    json.dumps(list(cheat_reasons or [])), diagnosis or "", gate_fail or "",
                    json.dumps(list(failing_checks or [])), now, now,
                ),
            )
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            return int(row["id"]) if row else None

    def find_verified_solution(
        self,
        *,
        user_id: str,
        task: str,
        min_similarity: float = 0.90,
        max_age_seconds: Optional[float] = None,
        limit: int = 200,
    ) -> Optional[Dict[str, Any]]:
        """Return the best near-identical, VERIFIED prior solution for this user to ADAPT as a
        starting point, or None. Never returns an unverified/failed run, and applies the same
        `unsafe_to_reuse` guard (swaps / identifier changes / polarity flips) the answer cache uses,
        so a different-but-similar task is not reused. The caller still re-verifies the seeded code
        through the full gate stack — this only seeds the first attempt, it never bypasses a gate."""
        now = time.time()
        user = user_id or "local"
        cutoff = None if max_age_seconds is None else now - max_age_seconds
        params: List[Any] = [user]
        sql = ("SELECT id, task, code, verification, updated_at, reuse_count "
               "FROM agent_runs WHERE user_id = ? AND verified = 1 AND code IS NOT NULL AND code <> ''")
        if cutoff is not None:
            sql += " AND updated_at >= ?"
            params.append(cutoff)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        best: Optional[Dict[str, Any]] = None
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        for row in rows:
            if unsafe_to_reuse(task, row["task"]):
                continue
            lex = question_similarity(task, row["task"])
            if lex < min_similarity:
                continue
            if best is None or lex > best["similarity"]:
                best = {"id": int(row["id"]), "task": row["task"], "code": row["code"],
                        "verification": row["verification"], "similarity": float(lex)}
        return best

    def record_agent_run_reuse(self, run_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE agent_runs SET reuse_count = reuse_count + 1, last_used_at = ? WHERE id = ?",
                (time.time(), int(run_id)),
            )

    def agent_failure_patterns(
        self,
        *,
        user_id: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """Aggregate recorded NON-verified runs into recurring failure patterns for a developer to
        review (read-only — the system changes nothing). Returns overall totals plus, per pattern, a
        count and a few example tasks. Verified runs are counted but never flagged as a pattern."""
        now = time.time()
        cutoff = None if max_age_seconds is None else now - max_age_seconds
        clauses: List[str] = []
        params: List[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if cutoff is not None:
            clauses.append("created_at >= ?")
            params.append(cutoff)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT task, verification, verified, stop_reason, cheat_reasons_json, diagnosis, "
                "gate_fail, failing_checks_json, hidden_passed, hidden_total FROM agent_runs"
                + where + " ORDER BY created_at DESC LIMIT ?",
                tuple(params) + (max(1, int(limit)),),
            ).fetchall()

        total = len(rows)
        verified_n = sum(1 for r in rows if r["verified"])
        buckets: Dict[str, Dict[str, Any]] = {}

        def _bump(key: str, label: str, task: str, note: str) -> None:
            b = buckets.setdefault(key, {"pattern": label, "count": 0, "examples": []})
            b["count"] += 1
            if len(b["examples"]) < 5:
                b["examples"].append({"task": (task or "")[:140], "note": (note or "")[:240]})

        def _loads(s: str) -> List[Any]:
            try:
                return list(json.loads(s or "[]"))
            except Exception:
                return []

        for r in rows:
            if r["verified"]:
                continue
            cheats = _loads(r["cheat_reasons_json"])
            fails = [str(f) for f in _loads(r["failing_checks_json"])]
            diag = (r["diagnosis"] or "").lower()
            gate = r["gate_fail"] or ""
            matched = False
            if any(("mask" in c.lower() or "renormalis" in c.lower() or "clamp" in c.lower())
                   for c in cheats):
                _bump("masking", "reward-hacking: masking (forced a check to pass)", r["task"],
                      "; ".join(cheats)); matched = True
            elif cheats:
                _bump("gaming", "reward-hacking: test-gaming / hardcoding", r["task"],
                      "; ".join(cheats)); matched = True
            if gate:
                _bump("delivery", "missing / empty output (delivery gate)", r["task"], gate)
                matched = True
            if any(f.startswith("test_definition") for f in fails):
                _bump("definition", "wrong reported quantity (definition gate)", r["task"],
                      ", ".join(fails)); matched = True
            if ("shape" in diag or "structure" in diag or "contract" in diag or "arity" in diag):
                _bump("contract", "input-rigidity / wrong shape-or-type (return-contract)",
                      r["task"], r["diagnosis"] or ""); matched = True
            if ("tolerance" in diag or "stochastic" in diag or "standard error" in diag):
                _bump("tolerance", "false-failure from a fixed tolerance on stochastic code",
                      r["task"], r["diagnosis"] or ""); matched = True
            if (r["hidden_total"] or 0) and (r["hidden_passed"] or 0) < (r["hidden_total"] or 0) \
                    and not matched:
                _bump("overfit", "fails on unseen inputs (overfit to the examples)", r["task"],
                      r["diagnosis"] or ""); matched = True
            if r["stop_reason"] in ("stall", "max_attempts") and not matched:
                _bump("stall", "stalled / hit the attempt cap without verifying", r["task"],
                      r["diagnosis"] or ""); matched = True
            if not matched:
                _bump("other", "other unverified outcome", r["task"],
                      (r["verification"] or "") + " " + (r["diagnosis"] or ""))

        patterns = sorted(buckets.values(), key=lambda b: b["count"], reverse=True)
        return {"total_runs": total, "verified": verified_n,
                "unverified": total - verified_n, "patterns": patterns}

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

    def relevant_facts(self, session_id: str, query: str,
                       limit: int = 6) -> List[Dict[str, Any]]:
        """Facts (global + this session) ranked by lexical overlap with the question, so only
        the ones likely relevant to THIS turn are injected — not the whole table. No embeddings
        or API calls (reuses question_tokens). Returns [{key, value, scope}] best-first."""
        q_toks = set(question_tokens(query or ""))
        if not q_toks:
            return []
        scored: List[tuple] = []
        for scope, sid in (("global", None), ("session", session_id)):
            if scope == "session" and not session_id:
                continue
            for r in self.list_facts(scope, sid):
                f_toks = set(question_tokens(f"{r['key']} {r['value']}"))
                overlap = len(q_toks & f_toks)
                if overlap > 0:
                    scored.append((overlap, float(r.get("updated_at") or 0.0), scope, r))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [{"key": r["key"], "value": r["value"], "scope": scope}
                for _ov, _ts, scope, r in scored[: max(0, int(limit))]]

    # ------- Compact-memory rolling summary --------------------------
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """The persisted rolling summary of OLDER turns + how many older turns it covers."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT mem_summary, mem_summary_upto, mem_summary_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return {"summary": "", "upto": 0, "at": None}
        return {"summary": row["mem_summary"] or "",
                "upto": int(row["mem_summary_upto"] or 0),
                "at": row["mem_summary_at"]}

    def set_session_summary(self, session_id: str, summary: str, upto: int) -> None:
        """Persist the rolling summary (and how many older turns it now covers) in conversations.db."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET mem_summary = ?, mem_summary_upto = ?, mem_summary_at = ? "
                "WHERE id = ?",
                (summary or "", int(upto), time.time(), session_id),
            )

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
