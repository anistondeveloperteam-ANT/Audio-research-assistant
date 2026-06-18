"""Performance/latency fixes for the agentic chat path (webapp/chat_logic.py +
backend/answering/agentic_answer.py). Fully offline — retrieval, the provider, the verifier,
and the code agent are all mocked, so nothing here touches the network, Oracle, or Docker.

Covers:
  P1  the runnable-Python simulation check is gated behind code intent (non-code queries skip
      it entirely, even when the draft happens to contain a ```python block); code-intent
      queries take the dedicated code/execution path instead.
  P2  the verify->rewrite loop early-stops on a passing verdict or one with no concrete gap,
      and never exceeds DEEP_MAX_LOOPS; has_concrete_gap classifies verdicts correctly.
  P2  bounded, rate-limit-safe concurrency: _gather_pass honors AGENT_PARALLELISM, the external
      evidence memo skips a re-fetch of the same (query, k), and a provider 429 degrades to a
      warning rather than crashing the request.
"""
import threading
import time

import pytest

import webapp.chat_logic as cl
from backend.answering.agentic_answer import (
    DEFAULT_FEEDBACK, has_actionable_feedback, has_concrete_gap, max_deep_loops)
from backend.answering.task_classifier import TaskClass
from backend.memory.store import MemoryStore
from backend.observability import tracing


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------
class _FakeProvider:
    is_available = True
    model = "fake"

    def __init__(self, answer="MVDR beamforming reduces noise by spatial filtering [1]."):
        self._answer = answer

    def stream_chat(self, messages, system=None, max_tokens=None, temperature=0.0,
                    yield_reasoning=False):
        yield self._answer


def _ext_item(title="WebResult"):
    return {"source_type": "web", "title": title, "text": "external passage about the topic",
            "url": "http://example.com/" + title}


def _set_task(monkeypatch, code_task, task_type):
    """Pin the router verdict (chat_logic imports classify locally, so patch the source module)."""
    monkeypatch.setattr("backend.answering.task_classifier.classify",
                        lambda q: TaskClass(code_task, task_type, 0.9, "test"))


def _drive(monkeypatch, tmp_path, *, code_task, task_type, provider, verify,
           verify_rounds=1, deep_loops=1, follow_up_items=None):
    """Run stream_chat_events offline and return (events, counters).

    Local RAG is OFF so the request takes the legacy external-only sweep into the agentic loop;
    the simulation check and the code agent are replaced by counters."""
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "false")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])

    _set_task(monkeypatch, code_task, task_type)

    counters = {"sim": 0, "code_agent": 0, "external": 0, "verify": 0}

    fu = follow_up_items if follow_up_items is not None else [_ext_item("FollowUp")]

    def fake_external(q, k):
        counters["external"] += 1
        # First gather populates the loop; later (follow-up) gathers add a fresh source so the
        # loop has something new to merge when a verdict reports a concrete gap.
        return ([_ext_item("Seed")] if counters["external"] == 1 else list(fu)), []

    monkeypatch.setattr(cl, "_gather_external_items", fake_external)

    def fake_sim(answer):
        counters["sim"] += 1
        return None                                  # never actually run Docker in a test

    monkeypatch.setattr(cl, "run_best_python_block", fake_sim)

    def fake_code_agent(*a, **k):
        counters["code_agent"] += 1
        yield {"type": "status", "message": "code agent ran"}

    monkeypatch.setattr(cl, "_run_code_agent", fake_code_agent)

    def counting_verify(provider, *, question, evidence, answer, run_info):
        counters["verify"] += 1
        return verify(counters["verify"])

    monkeypatch.setattr(cl, "verify_answer", counting_verify)
    monkeypatch.setattr(cl, "agentic_loop_enabled", lambda: True)
    monkeypatch.setattr(cl, "auto_review_enabled", lambda: False)
    monkeypatch.setattr(cl, "max_verify_rounds", lambda: verify_rounds)
    monkeypatch.setattr(cl, "max_deep_loops", lambda: deep_loops)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: provider)

    events = list(cl.stream_chat_events(sid, "How does MVDR beamforming reduce noise?"))
    return events, counters


def _passing(_round):
    return {"ok": True, "score": 95, "needs_more_search": False, "feedback": ""}


# ======================================================================
# Date awareness + entity attribution (so 'latest this year' isn't answered with old years,
# and a source about another org isn't credited to the asked-about one)
# ======================================================================
def test_today_note_anchors_the_real_current_date():
    import datetime as _dt
    note = cl._today_note()
    assert str(_dt.date.today().year) in note          # the live year, not a training-era default
    low = note.lower()
    assert "this year" in low and "latest" in low and "today" in low


def test_system_prompt_has_entity_attribution_rule():
    assert "ATTRIBUTION" in cl.SYSTEM_PROMPT
    assert "different entity" in cl.SYSTEM_PROMPT.lower()


def test_freshness_note_only_for_time_sensitive_queries():
    import datetime as _dt
    assert cl._freshness_note("explain how a Kalman filter works") == ""
    note = cl._freshness_note("what is the latest on OpenAI this year")
    assert note and str(_dt.date.today().year) in note


def test_freshness_query_goes_web_only_and_anchors_the_year(tmp_path, monkeypatch):
    # A 'latest ... this year' question must NOT be answered from the static local PDF library; it
    # goes web-only and the external search is anchored to the current year.
    import datetime as _dt
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "true")     # local ON, but freshness should skip it
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setenv("CRAG_ENABLED", "true")
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr("backend.answering.task_classifier.classify",
                        lambda q: TaskClass(False, "none", 0.9, "t"))
    local_calls = {"n": 0}
    monkeypatch.setattr(cl, "_gather_local_items",
                        lambda q, mode: (local_calls.__setitem__("n", local_calls["n"] + 1), ([], []))[1])
    ext_seen = {"q": None}
    monkeypatch.setattr(cl, "_gather_external_items",
                        lambda q, k: (ext_seen.__setitem__("q", q), ([_ext_item("News")], []))[1])
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    monkeypatch.setattr(cl, "agentic_loop_enabled", lambda: True)
    monkeypatch.setattr(cl, "auto_review_enabled", lambda: False)
    monkeypatch.setattr(cl, "max_verify_rounds", lambda: 1)
    monkeypatch.setattr(cl, "run_best_python_block", lambda a: None)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(cl, "verify_answer",
                        lambda provider, **kw: {"ok": True, "score": 95, "needs_more_search": False})

    list(cl.stream_chat_events(sid, "what is the latest on OpenAI this year"))

    assert local_calls["n"] == 0                        # static local library skipped for freshness
    assert ext_seen["q"] and str(_dt.date.today().year) in ext_seen["q"]   # search anchored to year


# ======================================================================
# P1 — simulation check is gated behind code intent
# ======================================================================
def test_noncode_query_skips_simulation_check_even_when_answer_has_python(tmp_path, monkeypatch):
    # A research answer that happens to include an illustrative code block must NOT spin up the
    # runnable-Python path — that was wasted Docker work on every loop for prose questions.
    answer_with_code = "Here is the idea [1]:\n\n```python\nprint('demo')\n```\nThat's all."
    events, counters = _drive(
        monkeypatch, tmp_path, code_task=False, task_type="none",
        provider=_FakeProvider(answer_with_code), verify=_passing)

    assert counters["sim"] == 0           # simulation check skipped for a non-code query
    assert counters["code_agent"] == 0    # and it never routed to the code agent
    statuses = " ".join(e.get("message", "") for e in events if e["type"] == "status").lower()
    assert "runnable python simulation" not in statuses
    assert any(e["type"] == "done" for e in events)


@pytest.mark.parametrize("task_type", ["deterministic", "simulation", "numeric_algorithm"])
def test_code_intent_takes_code_execution_path(tmp_path, monkeypatch, task_type):
    # Any code-intent task_type routes to the dedicated code agent (which runs/sandboxes the code)
    # BEFORE the prose loop — so the inline prose simulation check is never the one doing it.
    events, counters = _drive(
        monkeypatch, tmp_path, code_task=True, task_type=task_type,
        provider=_FakeProvider(), verify=_passing)

    assert counters["code_agent"] == 1    # code/execution path ran
    assert counters["sim"] == 0           # not via the prose loop's inline check
    assert counters["verify"] == 0        # prose verify loop was never entered


# ======================================================================
# P2 — has_concrete_gap classifies verdicts
# ======================================================================
def test_has_concrete_gap_true_only_for_actionable_verdicts():
    assert has_concrete_gap({"missing_evidence": ["the SNR figure"]}) is True
    assert has_concrete_gap({"citation_issues": ["[3] not in sources"]}) is True
    assert has_concrete_gap({"needs_more_search": True}) is True
    assert has_concrete_gap({"followup_query": "MVDR null depth"}) is True
    # A merely sub-threshold score with nothing actionable is NOT a concrete gap.
    assert has_concrete_gap({"ok": False, "score": 60}) is False
    assert has_concrete_gap({"missing_evidence": [], "citation_issues": [],
                             "needs_more_search": False, "followup_query": ""}) is False
    assert has_concrete_gap({}) is False


def test_has_actionable_feedback_distinguishes_specific_from_empty():
    assert has_actionable_feedback({"feedback": "soften the overstated claim about [2]"}) is True
    assert has_actionable_feedback({"feedback": "   "}) is False        # whitespace only
    assert has_actionable_feedback({"feedback": DEFAULT_FEEDBACK}) is False   # placeholder, not real
    assert has_actionable_feedback({}) is False


# ======================================================================
# P2 — verify->rewrite loop early-stops and respects DEEP_MAX_LOOPS
# ======================================================================
def test_loop_early_stops_when_first_draft_passes(tmp_path, monkeypatch):
    # Draft passes on round 1 -> stop immediately, even though up to 3 loops were allowed.
    events, counters = _drive(
        monkeypatch, tmp_path, code_task=False, task_type="none",
        provider=_FakeProvider(), verify=_passing, verify_rounds=3, deep_loops=3)

    assert counters["verify"] == 1        # no needless second draft/verify
    assert counters["external"] == 1      # only the seed gather; no follow-up search


def test_loop_early_stops_when_no_gap_and_no_actionable_feedback(tmp_path, monkeypatch):
    # Verdict fails the bar, names NO structured gap, AND gives no actionable feedback -> finalize
    # immediately (a rewrite would only chase a vague target).
    def no_gap(_round):
        return {"ok": False, "score": 55, "needs_more_search": False, "followup_query": "",
                "missing_evidence": [], "citation_issues": [], "feedback": ""}

    events, counters = _drive(
        monkeypatch, tmp_path, code_task=False, task_type="none",
        provider=_FakeProvider(), verify=no_gap, verify_rounds=3, deep_loops=3)

    assert counters["verify"] == 1
    assert counters["external"] == 1      # no follow-up search when there's nothing to fix
    statuses = " ".join(e.get("message", "") for e in events if e["type"] == "status").lower()
    assert "no concrete gap left" in statuses


def test_loop_does_one_feedback_guided_rewrite_then_stops(tmp_path, monkeypatch):
    # A failing verdict with SPECIFIC prose feedback (but no structured gap) must NOT be dropped:
    # the loop does exactly ONE evidence-only rewrite to apply the fix, then finalizes (it does not
    # run all 3 rounds chasing the same vague verdict). Guards the audit's medium regression.
    def feedback_only(_round):
        return {"ok": False, "score": 78, "needs_more_search": False, "followup_query": "",
                "missing_evidence": [], "citation_issues": [],
                "feedback": "soften the overstated claim about [2]"}

    events, counters = _drive(
        monkeypatch, tmp_path, code_task=False, task_type="none",
        provider=_FakeProvider(), verify=feedback_only, verify_rounds=3, deep_loops=3)

    assert counters["verify"] == 2        # one guided rewrite happened (not dropped, not 3 rounds)
    assert counters["external"] == 1      # the rewrite uses existing evidence — no new search
    statuses = " ".join(e.get("message", "") for e in events if e["type"] == "status").lower()
    assert "specific fix" in statuses and "no concrete gap left" in statuses


def test_loop_respects_deep_max_loops_cap(tmp_path, monkeypatch):
    # Every verdict reports a concrete gap, so the loop WOULD keep going — but DEEP_MAX_LOOPS=2
    # caps it at 2 verify rounds even though max_verify_rounds=5.
    def always_gap(_round):
        return {"ok": False, "score": 40, "needs_more_search": True,
                "followup_query": "MVDR covariance estimation", "missing_evidence": ["a detail"],
                "citation_issues": [], "feedback": "needs more"}

    events, counters = _drive(
        monkeypatch, tmp_path, code_task=False, task_type="none",
        provider=_FakeProvider(), verify=always_gap, verify_rounds=5, deep_loops=2)

    assert counters["verify"] == 2        # capped at DEEP_MAX_LOOPS, not max_verify_rounds (5)
    assert counters["external"] >= 2      # it did keep searching across the rounds it ran


def test_max_deep_loops_reads_env_and_clamps(monkeypatch):
    monkeypatch.setenv("DEEP_MAX_LOOPS", "1")
    assert max_deep_loops() == 1
    monkeypatch.setenv("DEEP_MAX_LOOPS", "99")     # clamped to the hard cap
    assert max_deep_loops() == 5
    monkeypatch.setenv("DEEP_MAX_LOOPS", "oops")   # invalid -> default
    assert max_deep_loops() == 2
    monkeypatch.delenv("DEEP_MAX_LOOPS", raising=False)
    assert max_deep_loops() == 2


# ======================================================================
# P2 — bounded, rate-limit-safe concurrency
# ======================================================================
def test_gather_pass_concurrency_is_bounded_by_agent_parallelism(monkeypatch):
    monkeypatch.setenv("AGENT_PARALLELISM", "2")
    tr = tracing.start_trace("test")
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def gather(q, k):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.05)                            # hold the slot so overlap is observable
        with lock:
            live["now"] -= 1
        return ([{"source_type": "web", "title": q, "text": q, "url": "http://x/" + q}], [])

    queries = [f"q{i}" for i in range(6)]
    items, warnings, timed_out = cl._gather_pass(
        queries, gather, lambda i, q: 1, trace=tr, span_name="external_search")

    assert len(items) == 6                          # every angle ran and merged
    assert live["peak"] <= 2                        # never more than AGENT_PARALLELISM in flight
    assert live["peak"] >= 2                         # ...and it really did run them in parallel


def test_agent_parallelism_clamps():
    import os
    old = os.environ.get("AGENT_PARALLELISM")
    try:
        os.environ["AGENT_PARALLELISM"] = "0"       # floor at 1
        assert cl._agent_parallelism() == 1
        os.environ["AGENT_PARALLELISM"] = "100"     # cap at 8
        assert cl._agent_parallelism() == 8
        os.environ["AGENT_PARALLELISM"] = "bad"     # invalid -> default 4
        assert cl._agent_parallelism() == 4
    finally:
        if old is None:
            os.environ.pop("AGENT_PARALLELISM", None)
        else:
            os.environ["AGENT_PARALLELISM"] = old


def test_external_memo_skips_refetch_of_same_query(monkeypatch):
    monkeypatch.setenv("EXTERNAL_GATHER_CACHE_TTL", "120")
    cl._EXT_CACHE.clear()
    calls = {"n": 0}

    class _Src:
        def __init__(self, title):
            self.title = title
            self.text = "evidence text"
        def to_public(self):
            return {"source_type": "web", "title": self.title, "url": "http://x/" + self.title}

    def fake_gather(query, max_results=20):
        calls["n"] += 1
        return [_Src("R1"), _Src("R2")], []

    monkeypatch.setattr(cl, "gather_external_evidence", fake_gather)

    first, _w1 = cl._gather_external_items("MVDR beamformer null depth", 8)
    second, _w2 = cl._gather_external_items("mvdr   beamformer   NULL depth", 8)   # same after normalize

    assert calls["n"] == 1                          # the second (query, k) was served from the memo
    assert [s["title"] for s in first] == [s["title"] for s in second]
    cl._EXT_CACHE.clear()


def test_external_memo_bypassed_for_freshness_sensitive_queries(monkeypatch):
    # 'latest/current/...' queries must always re-search — never served from the memo (matches the
    # answer cache's freshness policy so a stale 'latest' answer can't be built from old evidence).
    monkeypatch.setenv("EXTERNAL_GATHER_CACHE_TTL", "120")
    monkeypatch.delenv("ANSWER_CACHE_ALLOW_FRESHNESS_QUERIES", raising=False)
    cl._EXT_CACHE.clear()
    calls = {"n": 0}

    class _Src:
        title = "R1"
        text = "evidence"
        def to_public(self):
            return {"source_type": "web", "title": self.title, "url": "http://x/R1"}

    def fake_gather(query, max_results=20):
        calls["n"] += 1
        return [_Src()], []

    monkeypatch.setattr(cl, "gather_external_evidence", fake_gather)

    cl._gather_external_items("latest beamforming research", 8)
    cl._gather_external_items("latest beamforming research", 8)

    assert calls["n"] == 2                           # re-searched both times; memo never used
    assert not cl._EXT_CACHE                          # and nothing was stored for a fresh query
    cl._EXT_CACHE.clear()


def test_external_gather_429_degrades_to_warning(monkeypatch):
    cl._EXT_CACHE.clear()

    def rate_limited(query, max_results=20):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(cl, "gather_external_evidence", rate_limited)

    items, warnings = cl._gather_external_items("anything", 8)   # must NOT raise

    assert items == []
    assert warnings and "429" in warnings[0]
    cl._EXT_CACHE.clear()
