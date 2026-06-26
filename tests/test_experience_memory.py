"""EXPERIENCE MEMORY — the agent LEARNS day by day from its own runs (Phase 1).

When the pipeline corrects itself (an arithmetic override, a conclusion-matches-work reconcile) or the
user regenerates to a better answer, a short GENERALISABLE lesson is captured; on a future SIMILAR
question the top lessons are recalled (relevance x recency x confidence) and injected into the draft,
so the agent stops repeating mistakes and matches what the user prefers. Lessons that yield a verified
answer are reinforced; the table is bounded. No model training; deterministic; fail-open; gated by
EXPERIENCE_MEMORY (off suite-wide in conftest — this file opts in).

Proves: (a) a lesson generalises to a similar question even when numbers differ, but not to an
unrelated one; (b) semantic recall via a matching embedding; (c) reinforcement raises a lesson's rank;
(d) the table is pruned/bounded; (e) deterministic distillation (mistake + shape-only preference, no
content leak); (f) capture writes the right lessons (and nothing when gated off); (g) the live answer
path CAPTURES a mistake-lesson on a self-correction and RECALLS it into a later similar draft.
"""
import pytest

import webapp.chat_logic as cl
from backend.answering import experience as ex
from backend.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("EXPERIENCE_MEMORY", "true")     # conftest disables it suite-wide


def _mem(tmp_path):
    return MemoryStore(tmp_path / "m.db")


# ---- (a) lessons generalise to SIMILAR questions (incl. different numbers), not unrelated ones ----
def test_lesson_recalled_on_similar_question_even_with_a_different_number(tmp_path):
    m = _mem(tmp_path)
    m.record_lesson(user_id="u", kind="mistake",
                    question="compute the storage for a 3 minute audio file",
                    content="compute it in code")
    hit = m.recall_lessons(user_id="u", question="compute the storage for a 7 minute audio file")
    assert len(hit) == 1 and hit[0]["content"] == "compute it in code"      # generalised across 3 -> 7
    assert m.recall_lessons(user_id="u", question="who wrote Romeo and Juliet") == []   # not unrelated


# ---- (a2) a word-overlapping but DIFFERENT-INTENT question must NOT pull a lesson ----
def test_lesson_not_recalled_for_word_overlapping_different_intent(tmp_path):
    m = _mem(tmp_path)
    m.record_lesson(user_id="u", kind="mistake",
                    question="how long is a 3 minute audio file",
                    content="compute it in code")
    # shares "how long is ... audio ..." but asks about docs, not a duration calc -> must stay below the
    # relevance floor so a calc 'compute in code' lesson can't poison an unrelated-intent question.
    assert m.recall_lessons(user_id="u", question="how long is the audio documentation") == []


# ---- (b) semantic recall when a comparable embedding is available ----
def test_semantic_recall_via_matching_embedding(tmp_path):
    m = _mem(tmp_path)
    m.record_lesson(user_id="u", kind="preference", question="explain beamforming",
                    content="the user prefers a detailed answer", embedding=[1.0, 0.0, 0.0],
                    embedding_meta="gemini")
    # totally different WORDS, but a near-identical embedding + same meta -> recalled semantically
    got = m.recall_lessons(user_id="u", question="zzz qqq www", query_embedding=[0.98, 0.02, 0.0],
                           query_meta="gemini")
    assert len(got) == 1 and got[0]["kind"] == "preference"
    # a different provider/model tag is NOT comparable -> falls back to (failing) lexical -> nothing
    assert m.recall_lessons(user_id="u", question="zzz qqq www", query_embedding=[0.98, 0.02, 0.0],
                            query_meta="other") == []


# ---- (c) reinforcement makes a proven lesson outrank a fresher unproven one ----
def test_reinforcement_raises_rank_and_confidence(tmp_path):
    m = _mem(tmp_path)
    a = m.record_lesson(user_id="u", kind="mistake", question="estimate the bandwidth of a link",
                        content="A: check units")
    b = m.record_lesson(user_id="u", kind="preference", question="estimate the bandwidth of a link",
                        content="B: prefer concise")
    for _ in range(4):
        m.reinforce_lessons([a])                        # A repeatedly helped -> stronger
    ranked = m.recall_lessons(user_id="u", question="estimate the bandwidth of a network link", top_k=2)
    assert [r["content"] for r in ranked][0] == "A: check units"            # reinforced A ranks first
    assert next(r for r in ranked if r["id"] == a)["confidence"] > 1.0
    assert b is not None


# ---- (d) the table stays bounded (weakest evicted first) ----
def test_prune_keeps_table_bounded(tmp_path):
    m = _mem(tmp_path)
    for i in range(8):
        m.record_lesson(user_id="u", kind="mistake", question=f"unique question number {i} here",
                        content=f"lesson {i}")
    assert m.prune_lessons(user_id="u", max_per_user=5) >= 0
    m.prune_lessons(user_id="u", max_per_user=5)
    with m._conn() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM lessons WHERE user_id = 'u'").fetchone()["c"]
    assert n <= 5


def test_record_dedupes_per_question_and_kind(tmp_path):
    m = _mem(tmp_path)
    m.record_lesson(user_id="u", kind="mistake", question="same question", content="first")
    m.record_lesson(user_id="u", kind="mistake", question="same question", content="second")
    hits = m.recall_lessons(user_id="u", question="same question")
    assert len(hits) == 1 and hits[0]["content"] == "second"                # upgraded, not duplicated


# ---- (e) deterministic distillation: mistake from signals; preference is SHAPE-ONLY (no leak) ----
def test_lesson_from_outcome_signals():
    assert ex.lesson_from_outcome("q", corrections=[("100", "144")]) and \
        "compute" in ex.lesson_from_outcome("q", corrections=[("1", "2")]).lower()
    assert "derivation" in (ex.lesson_from_outcome("q", reconciled=True) or "")
    assert "verification" in (ex.lesson_from_outcome("q", rewritten=True) or "")
    assert ex.lesson_from_outcome("q") is None                              # nothing corrected -> no lesson


def test_preference_lesson_is_shape_only_never_leaks_content():
    secret = "the launch code is hunter2 and the ceo is alice"
    answer = secret + "\n\n## Details\n" + ("x " * 900) + "\n```py\nprint(1)\n```"
    lesson = ex.preference_lesson("q", answer)
    assert lesson and "detailed" in lesson and "runnable code" in lesson and "sections" in lesson
    assert "hunter2" not in lesson and "alice" not in lesson and "launch code" not in lesson
    assert ex.preference_lesson("q", "too short") is None


# ---- (f) capture writes the right lessons, and nothing when gated off ----
def test_capture_writes_mistake_and_preference(tmp_path):
    m = _mem(tmp_path)
    ex.capture_outcome(m, user_id="u", question="area of a 12 by 8 plot", answer="96",
                       corrections=[("100", "96")], verified=True)
    ex.capture_outcome(m, user_id="u", question="explain attention in transformers",
                       answer="A long, detailed, well-sectioned answer. " * 60,
                       regenerated=True, verified=True)
    kinds = {r["kind"] for r in m.recall_lessons(user_id="u", question="area of a 12 by 8 plot")}
    assert "mistake" in kinds
    assert {r["kind"] for r in m.recall_lessons(user_id="u", question="explain attention in transformers")} == {"preference"}


def test_mistake_lesson_not_captured_from_unverified_run(tmp_path):
    m = _mem(tmp_path)
    # the pipeline self-corrected but the FINAL answer still failed verification -> a 'fix' that didn't
    # work; learning from it would teach a bad run, so NOTHING is stored.
    ex.capture_outcome(m, user_id="u", question="area of a 12 by 8 plot", answer="96",
                       corrections=[("100", "96")], verified=False)
    assert m.recall_lessons(user_id="u", question="area of a 12 by 8 plot") == []


def test_zero_confidence_lesson_does_not_resurrect_at_full_strength(tmp_path):
    m = _mem(tmp_path)
    weak = m.record_lesson(user_id="u", kind="mistake", question="estimate link bandwidth here",
                           content="weak", confidence=0.0)
    strong = m.record_lesson(user_id="u", kind="mistake", question="estimate link bandwidth here now",
                             content="strong", confidence=1.0)
    ranked = m.recall_lessons(user_id="u", question="estimate link bandwidth", top_k=2)
    # a 0.0-confidence lesson must score ~0 (not be treated as 1.0), so 'strong' outranks 'weak'.
    assert ranked and ranked[0]["content"] == "strong"
    weak_row = next((r for r in ranked if r["id"] == weak), None)
    assert weak_row is None or weak_row["score"] == 0.0
    assert strong is not None


def test_capture_noop_when_disabled(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    monkeypatch.setenv("EXPERIENCE_MEMORY", "false")
    ex.capture_outcome(m, user_id="u", question="area of a 12 by 8 plot", answer="96",
                       corrections=[("100", "96")], verified=True)
    assert m.recall_lessons(user_id="u", question="area of a 12 by 8 plot") == []
    assert ex.recall(m, user_id="u", question="anything") == ("", [])


# ======================================================================================
# (g) END-TO-END on the live answer path: a self-correction is LEARNED and later RECALLED.
# ======================================================================================
class _Trace:
    def set(self, **k):
        return self

    def end(self):
        pass


class _P:
    """Reasoning draft states a FALSE equality (12 x 12 = 100) -> verify_calculation corrects it to
    144, so the run records a 'mistake' lesson. The quality judge passes it."""
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        if "answer-quality judge" in (system or "").lower():
            return ['{"ok": true, "score": 95}']
        return ["The area of a 12 by 12 square is 12 x 12 = 100 square units, a simple product."]

    def unavailable_message(self):
        return "n/a"


def test_live_path_captures_a_mistake_lesson_and_recalls_it(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, "area of a 12 by 12 square?")
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _P())

    done = [e for e in cl._reasoning_fallback("area of a 12 by 12 square?", mem, sid,
            qid["turn_id"], qid["node_id"], "local", False, None, None, _Trace())
            if e["type"] == "done"][0]
    assert "12 x 12 = 144" in done["answer"]                 # the arithmetic was corrected (the mistake)

    # The run LEARNED a mistake-lesson; it is recalled for a SIMILAR (different-number) question and
    # injected into the next draft's prompt.
    lessons = mem.recall_lessons(user_id="local", question="area of a 9 by 9 square?")
    assert lessons and lessons[0]["kind"] == "mistake"
    block = cl._build_compact_context(mem, sid, "area of a 9 by 9 square?", user_id="local")["system_extra"]
    assert "LEARNED FROM EXPERIENCE" in block
