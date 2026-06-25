"""EFFORT SCALES TO THE QUESTION + the planner stays FAITHFUL to what was asked.

The research pipeline used to do MAXIMUM work on every question (many planned angles x many source
channels x repeated verify/refine loops with broad re-searches), and its planner could DISTORT the
question while decomposing it (e.g. rewrite a current-year question into a multi-year past range).
These tests prove the fix, GENERAL across question types/domains — not tied to any one topic:

  (a) decomposition PRESERVES the user's scope/time-frame/intent — a simple scoped question is NOT
      broadened into extra angles, and the planner prompt mandates scope/time-frame fidelity;
  (b) a simple single-intent question gets a MINIMAL budget (0 angles, 1 verify pass) and _deep_queries
      returns just the one query — sharply fewer searches than the complex path; caps hold;
  (c) verify loops do NOT re-run searches for a simple question (a failing verdict that WANTS another
      search is denied), while a complex question is allowed a single BOUNDED gap-search;
  (d) effort scales UP only for genuinely complex / multi-part questions;
  (e) the relevance gate drops sources that fall outside the question's subject/scope (never cited).

Deterministic + offline: the gauge is pure; the integration cases mock provider / memory / search so
the ONLY thing that varies the search count is the effort budget.
"""
import webapp.chat_logic as cl
from backend.answering.effort import Effort, assess_effort, is_complex
from backend.answering.relevance_gate import _RELEVANCE_SYSTEM, relevant_source_indices
from backend.agent.research_agent import _PLAN_SYSTEM
from backend.memory.store import MemoryStore


# Representative questions across DOMAINS — the fix is general, never topic-specific.
_SIMPLE = [
    "What is the capital of France?",
    "Define entropy.",
    "What was the population of Tokyo in 2010?",
    "Who wrote Hamlet?",
]
_COMPLEX = [
    "Compare the advantages and disadvantages of CNNs versus transformers for vision.",
    "List the main types of database indexes and explain how each works.",
    "What is X? Why does it matter? How is it measured?",                       # multiple questions
    "Walk me through a comprehensive, in-depth analysis of the topic at hand.",  # explicit depth
]


# ---- (b) / (d) the gauge: simple => minimal, complex => scaled, caps ALWAYS hold ----------------
def test_b_simple_question_gets_minimal_budget():
    for q in _SIMPLE:
        assert assess_effort(q, angle_cap=3, loop_cap=2) == Effort(0, 1, "simple"), q


def test_d_complex_question_scales_up_within_caps():
    for q in _COMPLEX:
        e = assess_effort(q, angle_cap=3, loop_cap=2)
        assert e.label == "complex" and e.angles == 3 and e.max_loops == 2, q


def test_gauge_never_exceeds_caps():
    e = assess_effort(_COMPLEX[0], angle_cap=5, loop_cap=4)
    assert e.angles <= 5 and e.max_loops <= 4              # bounded by the caps the caller passes
    e0 = assess_effort(_COMPLEX[0], angle_cap=0, loop_cap=1)
    assert e0.angles == 0 and e0.max_loops == 1            # a 0/low cap is still honoured


def test_is_complex_classifies_general_questions():
    assert all(not is_complex(q) for q in _SIMPLE)
    assert all(is_complex(q) for q in _COMPLEX)


def test_complex_signals_multiple_parts_or_length():
    assert is_complex("Explain A and discuss B and analyse C and summarise D")   # >= 2 ' and '
    assert is_complex("word " * 31)                                              # long, detailed ask
    assert not is_complex("What is the boiling point of water and is it 100C?")  # a single ' and '


# ---- (a) decomposition preserves the user's scope / time-frame (no broadening) -------------------
def test_a_simple_scoped_question_is_not_broadened(monkeypatch):
    # A simple, scoped question must NOT be decomposed: nothing is added, so its scope, time frame
    # and intent are preserved verbatim. With scaling on, _deep_queries returns just the literal query.
    monkeypatch.setenv("EFFORT_SCALING", "true")               # conftest disables it suite-wide
    scoped = "What was the population of Tokyo in 2010?"
    assert assess_effort(scoped, angle_cap=3, loop_cap=2).angles == 0
    assert cl._deep_queries(scoped) == [scoped]                # unchanged, not broadened into angles


def test_a_planner_prompt_mandates_scope_fidelity():
    p = _PLAN_SYSTEM.lower()
    assert "preserve" in p and "time frame" in p                       # keeps the user's time frame
    assert "not broaden" in p                                          # must not broaden the question
    assert "current state" in p                                        # forbidden as an injected angle
    assert "genuinely multi-part" in p                                 # decompose only when warranted


# ---- (e) relevance gate drops out-of-scope / off-subject sources ---------------------------------
class _GateJudge:
    """A judge that keeps only the in-scope source (1) and drops the off-scope one (2)."""
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        return ['{"relevant": [1]}']

    def unavailable_message(self):
        return "n/a"


def test_e_relevance_gate_drops_out_of_scope_sources(monkeypatch):
    monkeypatch.setenv("SOURCE_RELEVANCE_GATE", "true")               # conftest disables it suite-wide
    items = [
        {"title": "On scope", "text": "directly answers the asked question within its scope"},
        {"title": "Off scope", "text": "same broad topic but a different year and a different entity"},
    ]
    keep = relevant_source_indices(_GateJudge(), question="a scoped question", items=items)
    assert keep == {1}                                                # the off-scope source is dropped


def test_e_relevance_prompt_enforces_scope_and_recency():
    s = _RELEVANCE_SYSTEM.lower()
    assert "scope" in s and "recency" in s
    assert "out-of-scope" in s or "outside it" in s


# ---- (c) verify loops re-search only for genuinely complex questions, and only BOUNDEDLY ----------
class _DraftP:
    """Always returns a usable grounded draft; the verifier is mocked separately."""
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        return ["A grounded draft answer citing the evidence [1], long enough to be a real answer. " * 2]

    def unavailable_message(self):
        return "n/a"


# The planner returns these 3 fixed angles; only a COMPLEX question (effort.angles > 0) ever reaches
# it — a simple question is never planned, so these never appear in its searches.
_ANGLES = ["angle one about the topic", "angle two about the topic", "angle three about the topic"]
_PASS = {"ok": True, "score": 95}
_FAIL_WANTS_SEARCH = {"ok": False, "score": 20, "needs_more_search": True, "followup_query": "narrow gap query"}
_COMPLEX_Q = "Compare the pros and cons of X versus Y in depth."


def _loop_env(monkeypatch, mem, searches, verdict):
    """Drive the pipeline with everything mocked EXCEPT the effort gauge, recording every external
    search query into `searches` so a test can assert exactly what was searched and how often."""
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "false", "ENABLE_LOCAL_RAG": "false", "ENABLE_WEB_SEARCH": "true",
                 "CRAG_ENABLED": "true", "CODE_INTENT_SEMANTIC": "false",
                 "ENABLE_AGENTIC_ANSWER_LOOP": "true", "AGENTIC_INDEPENDENT_VERIFY": "false",
                 "SOURCE_RELEVANCE_GATE": "false", "EFFORT_SCALING": "true"}.items():   # opt back in
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _DraftP())
    monkeypatch.setattr(cl, "auto_review_enabled", lambda: False)   # deep mode forces it on; keep lean
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr("backend.agent.research_agent._plan", lambda provider, question: list(_ANGLES))
    monkeypatch.setattr(cl, "verify_answer", lambda *a, **k: dict(verdict))

    def ext(query, k):
        searches.append(query)
        return ([{"source_type": "web", "title": "S", "url": f"http://x/{len(searches)}",
                  "text": "some relevant evidence about the question", "score": 0.7}], [])

    monkeypatch.setattr(cl, "_gather_external_items", ext)


def _drive_deep(sid, q):
    # DEEP mode is the full-sweep profile (historically 3 angles + 3 verify rounds for EVERY question).
    for ev in cl.stream_chat_events(sid, q, mode="deep"):
        if ev["type"] in ("done", "error", "sanity"):
            return ev
    return None


def test_b_simple_question_minimal_search_even_in_deep_mode(tmp_path, monkeypatch):
    # The effort gauge scales a SIMPLE question DOWN even under the heavy DEEP profile: no angle
    # planning and a single verify pass — so even a 'needs more search' verdict yields ONE search.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    searches: list = []
    _loop_env(monkeypatch, mem, searches, _FAIL_WANTS_SEARCH)
    done = _drive_deep(sid, "What is the capital of France?")
    assert done and done["type"] == "done"
    assert searches == ["What is the capital of France?"]       # one search; no angles, no re-search


def test_d_complex_question_plans_angles_in_deep_mode(tmp_path, monkeypatch):
    # A genuinely complex question DOES get the multi-angle sweep (effort scales up). A passing verdict
    # stops the loop after one pass, isolating the INITIAL planning breadth.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    searches: list = []
    _loop_env(monkeypatch, mem, searches, _PASS)
    done = _drive_deep(sid, _COMPLEX_Q)
    assert done and done["type"] == "done"
    assert searches[0] == _COMPLEX_Q and searches[1:] == _ANGLES   # the question + its 3 planned angles
    assert len(searches) == 4                                      # sharply more than the simple path (1)


def test_c_verify_loop_uses_narrow_gap_searches_not_broad_resweeps(tmp_path, monkeypatch):
    # When a complex question's verify loop re-searches, each re-search is the SINGLE narrow gap query —
    # never a re-run of the broad multi-angle sweep — and the number of loop searches is bounded.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    searches: list = []
    _loop_env(monkeypatch, mem, searches, _FAIL_WANTS_SEARCH)
    done = _drive_deep(sid, _COMPLEX_Q)
    assert done and done["type"] == "done"
    assert searches[:4] == [_COMPLEX_Q, *_ANGLES]              # the broad angle sweep runs ONCE, up front
    assert searches[4:]                                        # the loop did re-search (complex => allowed)
    assert set(searches[4:]) == {"narrow gap query"}          # every loop search is the narrow gap only
    assert len(searches) <= 4 + 3                             # bounded by the deep loop cap (no runaway)
