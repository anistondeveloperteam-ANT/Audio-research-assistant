"""Conversation-aware follow-up routing + source-relevance display.

A follow-up that refers to the chat so far ("what is the output of the above code?", "explain that")
must be answered FROM the conversation — by re-running earlier code or from context — NOT by a fresh
web/paper sweep that pulls dozens of off-topic sources. A genuinely new question still searches.
The source panel shows only the sources the answer actually cited (so a maths answer never lists
biology hits). Fully offline — the router, classifier, provider, and retrieval are mocked.
"""
import pytest

import webapp.chat_logic as cl
from backend.answering.conversation_router import ConversationRoute, route
from backend.answering.task_classifier import TaskClass
from backend.memory.store import MemoryStore


# ======================================================================
# conversation_router — regex fallback (deterministic; CONVERSATION_ROUTER=false skips the LLM)
# ======================================================================
def test_router_no_history_is_always_research():
    assert route("what is the output of the above?", []).kind == "research"


def test_router_regex_code_output(monkeypatch):
    monkeypatch.setenv("CONVERSATION_ROUTER", "false")
    r = route("what is the output of the above code?", [{"role": "user", "content": "x"}])
    assert r.kind == "code_output"


def test_router_regex_context_followup(monkeypatch):
    monkeypatch.setenv("CONVERSATION_ROUTER", "false")
    r = route("explain the above code more simply", [{"role": "user", "content": "x"}])
    assert r.kind == "context"


def test_router_regex_standalone_is_research(monkeypatch):
    monkeypatch.setenv("CONVERSATION_ROUTER", "false")
    r = route("what is the Black-Scholes equation for option pricing",
              [{"role": "user", "content": "x"}])
    assert r.kind == "research"


@pytest.mark.parametrize("q", [
    "what is the output impedance of an op-amp?",   # 'the output' is a content word, not a back-ref
    "what is the result of the 2020 election?",
    "how does that algorithm work?",
    "what is its boiling point?",
    "what are the former soviet republics?",
])
def test_router_regex_does_not_misroute_new_questions(monkeypatch, q):
    # Self-contained NEW questions that merely contain pronouns/'output'/'result' must NOT be
    # treated as follow-ups by the regex fallback — they must still go to research (and search).
    monkeypatch.setenv("CONVERSATION_ROUTER", "false")
    assert route(q, [{"role": "user", "content": "x"}]).kind == "research"


def test_router_regex_confidence_is_below_floor(monkeypatch):
    # The regex verdict scores 0.5 — below the default 0.6 follow-up floor — so when the LLM router
    # is unavailable, no message is ever diverted away from search on the regex alone.
    monkeypatch.setenv("CONVERSATION_ROUTER", "false")
    r = route("run it again", [{"role": "user", "content": "x"}])
    assert r.kind == "code_output" and r.confidence < cl._followup_confidence_floor()


# ======================================================================
# Source-relevance display: keep only the sources the answer cited
# ======================================================================
def test_cited_source_numbers_parsing():
    assert cl._cited_source_numbers("uses [2] and [5][5], not [x]") == {2, 5}
    assert cl._cited_source_numbers("no citations") == set()
    # Grouped citations like [1, 3] must be recognized (they're dropped otherwise).
    assert cl._cited_source_numbers("supported by [1, 3] and also [7]") == {1, 3, 7}


def test_relevant_sources_keeps_only_cited(monkeypatch):
    monkeypatch.setenv("SOURCE_RELEVANCE_DISPLAY", "true")
    srcs = [{"n": 1, "title": "maths"}, {"n": 2, "title": "biology"}, {"n": 3, "title": "maths2"}]
    kept = cl._relevant_sources("The result is X [1], also [3].", srcs)
    assert {s["n"] for s in kept} == {1, 3}            # the off-topic [2] is dropped


def test_relevant_sources_falls_back_when_nothing_cited(monkeypatch):
    monkeypatch.setenv("SOURCE_RELEVANCE_DISPLAY", "true")
    srcs = [{"n": 1}, {"n": 2}]
    assert cl._relevant_sources("an answer with no [n] citations", srcs) == srcs


# ======================================================================
# Integration: a follow-up is answered from the conversation, never searched
# ======================================================================
def _seed_prior(mem, sid, answer_md):
    """Add one prior user question + assistant answer so the next message is a follow-up."""
    info = mem.start_question(sid, "write python for box ordering")
    mem.add_answer_version(info["turn_id"], answer_md)


def _common(monkeypatch, mem, *, route_kind, conf=0.9, resolved=None):
    monkeypatch.setattr(cl, "_memory", mem)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "false")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr("backend.answering.task_classifier.classify",
                        lambda q: TaskClass(False, "none", 0.9, "test"))
    monkeypatch.setattr("backend.answering.conversation_router.route",
                        lambda q, hist, **k: ConversationRoute(route_kind, resolved or q, conf, "test"))
    ext = {"n": 0}
    monkeypatch.setattr(cl, "_gather_external_items",
                        lambda q, k: (ext.__setitem__("n", ext["n"] + 1), ([_extsrc()], []))[1])
    return ext


class _FakeProvider:
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system=None, max_tokens=None, temperature=0.0,
                    yield_reasoning=False):
        yield "Answering directly from our conversation — no new search was needed."


def test_followup_code_output_reruns_prior_code_and_skips_search(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "Here is the code:\n\n```python\nprint(42)\n```\n")
    ext = _common(monkeypatch, mem, route_kind="code_output")
    monkeypatch.setattr(cl, "_run_prior_code", lambda code: f"Output: 42 (ran: {code.strip()})")

    events = list(cl.stream_chat_events(sid, "what is the output of above one?"))

    done = [e for e in events if e["type"] == "done"]
    assert done and "Output: 42" in done[0]["answer"]
    assert "print(42)" in done[0]["answer"]            # it found + ran the prior code block
    assert ext["n"] == 0                               # NO web/paper search
    srcs = [e for e in events if e["type"] == "sources"]
    assert srcs and srcs[-1]["sources"] == []          # no off-topic source pile


def test_followup_code_output_with_no_code_falls_back_to_conversation(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "It works by spatial filtering — no code here.")
    ext = _common(monkeypatch, mem, route_kind="code_output")
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _FakeProvider())

    events = list(cl.stream_chat_events(sid, "what is the output of above one?"))

    done = [e for e in events if e["type"] == "done"]
    assert done and "from our conversation" in done[0]["answer"]   # fell back to context answer
    assert ext["n"] == 0


def test_context_followup_answers_from_conversation_without_search(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "The box-ordering algorithm sorts by weight.")
    ext = _common(monkeypatch, mem, route_kind="context")
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _FakeProvider())

    events = list(cl.stream_chat_events(sid, "explain that again, simpler"))

    done = [e for e in events if e["type"] == "done"]
    assert done and "from our conversation" in done[0]["answer"]
    assert ext["n"] == 0                               # answered from context, no search
    srcs = [e for e in events if e["type"] == "sources"]
    assert srcs and srcs[-1]["sources"] == []


def test_python_blocks_in_order_preserves_document_order():
    from backend.answering.agentic_answer import python_blocks_in_order
    md = "```python\nA = 1\n```\nmid\n```python\nprint(A)\n```"
    assert python_blocks_in_order(md) == ["A = 1", "print(A)"]   # order, not longest-first


def test_code_output_runs_the_last_block_not_the_longest(tmp_path, monkeypatch):
    # A prior answer with a long helper demo followed by the short final program: 'the above code'
    # means the LAST/canonical block, not whichever block is longest.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(
        mem, sid,
        "First a long helper:\n\n```python\ndef helper():\n    return 'a very long demo block ' * 5\n```\n\n"
        "Then the final program:\n\n```python\nprint(2 + 2)\n```\n")
    _common(monkeypatch, mem, route_kind="code_output")
    seen = {"code": None}
    monkeypatch.setattr(cl, "_run_prior_code", lambda code: (seen.__setitem__("code", code), "ran")[1])

    list(cl.stream_chat_events(sid, "what is the output of the above code?"))

    assert seen["code"] is not None
    assert "print(2 + 2)" in seen["code"] and "helper" not in seen["code"]   # the LAST block


def _drive_research(monkeypatch, mem, sid, question, *, route_kind, conf, resolved):
    """Run the full research path with mocked retrieval/provider; return (ext_count, planned_query)."""
    ext = _common(monkeypatch, mem, route_kind=route_kind, conf=conf, resolved=resolved)
    planned = {"q": None}
    monkeypatch.setattr(cl, "_deep_queries",
                        lambda q: (planned.__setitem__("q", q), [q])[1])
    monkeypatch.setattr(cl, "agentic_loop_enabled", lambda: True)
    monkeypatch.setattr(cl, "auto_review_enabled", lambda: False)
    monkeypatch.setattr(cl, "max_verify_rounds", lambda: 1)
    monkeypatch.setattr(cl, "run_best_python_block", lambda a: None)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(cl, "verify_answer",
                        lambda provider, **kw: {"ok": True, "score": 95, "needs_more_search": False})
    events = list(cl.stream_chat_events(sid, question))
    assert any(e["type"] == "done" for e in events)
    return ext["n"], planned["q"]


def test_new_research_question_with_history_still_searches(tmp_path, monkeypatch):
    # A self-contained NEW question (no back-reference), even mid-conversation, must still run
    # retrieval (router says "research") AND keep the user's exact words — not a stray LLM rewrite.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "Earlier we discussed beamforming.")
    ext_n, planned_q = _drive_research(monkeypatch, mem, sid, "what is the wave equation",
                                       route_kind="research", conf=0.9, resolved="resolved physics query")
    assert ext_n >= 1                                  # it DID search
    assert planned_q == "what is the wave equation"    # standalone -> original query, not the rewrite


def test_research_followup_with_reference_uses_resolved_query(tmp_path, monkeypatch):
    # A research-class follow-up that DOES reference the chat ("...about that") searches with the
    # anaphora-resolved query so the right topic is retrieved.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "We discussed the MVDR beamformer.")
    ext_n, planned_q = _drive_research(monkeypatch, mem, sid, "tell me more about that approach",
                                       route_kind="research", conf=0.9, resolved="more about the MVDR beamformer")
    assert ext_n >= 1
    assert planned_q == "more about the MVDR beamformer"   # reference present -> resolved query


def test_confident_followup_verdict_vetoed_for_standalone_question(tmp_path, monkeypatch):
    # Defense in depth: even a CONFIDENT 'context' verdict cannot skip search for a self-contained
    # question with no back-reference — the deixis veto forces it back to research.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "Earlier we discussed beamforming.")
    ext_n, _ = _drive_research(monkeypatch, mem, sid, "explain quantum entanglement in detail",
                               route_kind="context", conf=0.95, resolved="explain quantum entanglement in detail")
    assert ext_n >= 1                                  # vetoed -> searched, despite high confidence


def test_plausibly_references_context_veto():
    assert cl._plausibly_references_context("explain the above code") is True
    assert cl._plausibly_references_context("make it simpler") is True
    assert cl._plausibly_references_context("and the time complexity?") is True
    assert cl._plausibly_references_context("why?") is True
    assert cl._plausibly_references_context("what is the output impedance of an op-amp") is False
    assert cl._plausibly_references_context("what is the result of the 2020 election") is False
    assert cl._plausibly_references_context("explain quantum entanglement in detail") is False


def test_cited_numbers_ignore_code_subscripts():
    # An array index inside a code block must NOT be counted as a citation.
    text = "See [2].\n\n```python\nx = arr[5]\nprint(arr[9])\n```\n\nAlso `arr[7]` inline."
    assert cl._cited_source_numbers(text) == {2}


def test_low_confidence_followup_still_searches(tmp_path, monkeypatch):
    # The router says "context" but with LOW confidence (< the 0.6 floor) -> the safety net treats
    # it as research and SEARCHES, rather than answering from chat on a shaky verdict.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "Earlier we discussed beamforming.")
    ext_n, _ = _drive_research(monkeypatch, mem, sid, "is this approach safe",
                               route_kind="context", conf=0.4, resolved="is this approach safe")
    assert ext_n >= 1                                  # low confidence -> still searched


def test_freshness_sensitive_followup_still_searches(tmp_path, monkeypatch):
    # A time-sensitive question ('latest ...') must always hit a fresh search, even if the router
    # confidently calls it a follow-up.
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _seed_prior(mem, sid, "Earlier we discussed beamforming.")
    ext_n, _ = _drive_research(monkeypatch, mem, sid, "what is the latest on that this week",
                               route_kind="context", conf=0.95, resolved="latest beamforming news")
    assert ext_n >= 1                                  # freshness overrides the follow-up divert


def _extsrc():
    return {"source_type": "web", "title": "Wave", "text": "the wave equation ...",
            "url": "http://example.com/wave"}
