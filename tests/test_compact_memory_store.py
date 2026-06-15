"""Store-level pieces of compact conversation memory: rolling-summary persistence (survives
reopen), relevance-ranked facts, and the token estimator. SQLite only — no network."""
from backend.memory.store import MemoryStore, estimate_tokens


def _mem(tmp_path):
    return MemoryStore(tmp_path / "memory.db", conversations_path=tmp_path / "conversations.db")


def test_estimate_tokens_is_roughly_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_session_summary_defaults_then_persists_and_survives_reopen(tmp_path):
    cache, conv = tmp_path / "memory.db", tmp_path / "conversations.db"
    mem = MemoryStore(cache, conversations_path=conv)
    sid = mem.create_session(user_id="anjan")

    got = mem.get_session_summary(sid)
    assert got == {"summary": "", "upto": 0, "at": None}     # fresh session: no summary yet

    mem.set_session_summary(sid, "User asked about MVDR; we covered the distortionless constraint.", 6)
    got = mem.get_session_summary(sid)
    assert got["summary"].startswith("User asked about MVDR")
    assert got["upto"] == 6 and got["at"] is not None

    reopened = MemoryStore(cache, conversations_path=conv)    # simulate a page refresh / new process
    again = reopened.get_session_summary(sid)
    assert again["summary"] == got["summary"] and again["upto"] == 6


def test_relevant_facts_ranks_by_overlap_and_filters_irrelevant(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    mem.upsert_fact("session", "beamformer", "user is implementing an MVDR beamformer", session_id=sid)
    mem.upsert_fact("session", "language", "user prefers Python with numpy", session_id=sid)
    mem.upsert_fact("global", "cooking", "user likes pasta recipes")

    hits = mem.relevant_facts(sid, "how does the MVDR beamformer suppress noise?", limit=6)
    keys = [h["key"] for h in hits]
    assert keys and keys[0] == "beamformer"                  # best lexical overlap first
    assert "cooking" not in keys                             # unrelated fact excluded

    # A query overlapping the Python fact surfaces it; a totally unrelated query yields nothing.
    assert any(h["key"] == "language" for h in mem.relevant_facts(sid, "numpy python tips", 6))
    assert mem.relevant_facts(sid, "weather forecast tomorrow", 6) == []


def test_relevant_facts_respects_limit(tmp_path):
    mem = _mem(tmp_path)
    sid = mem.create_session(user_id="anjan")
    for i in range(5):
        mem.upsert_fact("session", f"signal{i}", "about audio signal processing", session_id=sid)
    hits = mem.relevant_facts(sid, "audio signal processing question", limit=2)
    assert len(hits) == 2
