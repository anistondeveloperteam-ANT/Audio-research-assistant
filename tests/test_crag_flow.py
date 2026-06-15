"""CRAG retrieval flow in webapp/chat_logic.py: grade local PDF evidence, then act.

Fully offline — local/external retrieval and the deep-query planner are mocked, and we stop
consuming the event stream at the `sources` event (the grade + external decision is complete by
then, before any LLM generation)."""
import time
import types

import webapp.chat_logic as cl
from backend.memory.store import MemoryStore
from backend.observability import tracing

QUESTION = "How does MVDR beamforming reduce noise?"


def _local(score, title="MVDR Paper"):
    # Distinct text + pages per chunk so _extend_unique keeps them as separate evidence
    # (identical chunks would correctly dedupe to one, which is not what these tests probe).
    page = int(round(score * 100))
    return {"source_type": "local_pdf", "title": title, "section": "Method",
            "text": f"local PDF passage @{score} about the topic", "score": score,
            "page_start": page, "page_end": page + 1}


def _ext(title="WebResult"):
    return {"source_type": "web", "title": title, "text": "external passage",
            "url": "http://example.com/" + title}


def _drive(monkeypatch, tmp_path, local_items, *, web=True, crag=True):
    """Run the chat stream with mocked retrieval; return (events, external_calls, sources)."""
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "false")
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "true")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true" if web else "false")
    monkeypatch.setenv("CRAG_ENABLED", "true" if crag else "false")
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    monkeypatch.setattr(cl, "_gather_local_items", lambda q, mode: (list(local_items), []))

    external_calls = []

    def fake_external(q, k):
        external_calls.append((q, k))
        return ([_ext()], [])

    monkeypatch.setattr(cl, "_gather_external_items", fake_external)

    events, sources = [], []
    for ev in cl.stream_chat_events(sid, QUESTION):
        events.append(ev)
        if ev["type"] == "sources":
            sources = ev["sources"]
        if ev["type"] in ("sources", "done", "error", "sanity"):
            break
    return events, external_calls, sources


def _statuses(events):
    return " ".join(e.get("message", "") for e in events if e["type"] == "status").lower()


# ----------------------------------------------------------------------
# STRONG -> answer from PDFs, do NOT search externally
# ----------------------------------------------------------------------
def test_strong_grade_skips_external_search(tmp_path, monkeypatch):
    local = [_local(0.80), _local(0.72), _local(0.40)]
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=True)

    assert external_calls == []                       # the adaptive win: no external spend
    assert "strong match" in _statuses(events)
    titles = [s["title"] for s in sources]
    assert "MVDR Paper" in titles                     # answered from the PDFs
    assert "WebResult" not in titles


# ----------------------------------------------------------------------
# PARTIAL -> keep PDF evidence AND search externally
# ----------------------------------------------------------------------
def test_partial_grade_keeps_local_and_adds_external(tmp_path, monkeypatch):
    local = [_local(0.62), _local(0.34)]              # one strong (<count), one partial
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=True)

    assert external_calls, "external search should run on a PARTIAL grade"
    assert "partially covered" in _statuses(events)
    titles = [s["title"] for s in sources]
    assert "MVDR Paper" in titles and "WebResult" in titles   # merged


# ----------------------------------------------------------------------
# NONE -> drop local, go fully external
# ----------------------------------------------------------------------
def test_none_grade_drops_local_and_goes_external(tmp_path, monkeypatch):
    local = [_local(0.21), _local(0.10)]             # nothing clears the partial floor
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=True)

    assert external_calls, "external search should run on a NONE grade"
    assert "not in your pdfs" in _statuses(events)
    titles = [s["title"] for s in sources]
    assert "WebResult" in titles
    assert "MVDR Paper" not in titles                # local evidence discarded


# ----------------------------------------------------------------------
# PARTIAL with web search OFF -> degrade gracefully to the local evidence
# ----------------------------------------------------------------------
def test_partial_grade_without_web_uses_local_only(tmp_path, monkeypatch):
    local = [_local(0.62), _local(0.34)]
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=False)

    assert external_calls == []                      # web off -> nothing external to call
    assert "web search is off" in _statuses(events)
    assert [s["title"] for s in sources] == ["MVDR Paper", "MVDR Paper"]


# ----------------------------------------------------------------------
# NONE with web search OFF -> no sources at all (local dropped, nothing external)
# ----------------------------------------------------------------------
def test_none_grade_without_web_yields_no_sources(tmp_path, monkeypatch):
    local = [_local(0.21), _local(0.10)]             # nothing clears the partial floor -> NONE
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=False)

    assert external_calls == []                      # web off -> nothing external
    assert sources == []                             # NONE drops local; nothing left
    assert "web search is off" in _statuses(events)


# ----------------------------------------------------------------------
# CRAG disabled -> original concurrent sweep still runs (local + external together)
# ----------------------------------------------------------------------
def test_crag_disabled_uses_legacy_concurrent_sweep(tmp_path, monkeypatch):
    local = [_local(0.80), _local(0.72)]             # would be STRONG, but CRAG is off
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=True, crag=False)

    assert external_calls, "legacy sweep always runs external when web search is on"
    titles = [s["title"] for s in sources]
    assert "MVDR Paper" in titles and "WebResult" in titles


# ======================================================================
# Code-from-paper: a code-intent query whose algorithm is in the PDFs is
# implemented from the paper (cited), with GitHub refs only when thin.
# ======================================================================
CODE_Q = "write python code for the MVDR beamformer"


def _drive_code(monkeypatch, tmp_path, local_items, *, crag=True):
    """Run the code-intent route with mocked local retrieval + a captured run_agent."""
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "true")
    monkeypatch.setenv("CRAG_ENABLED", "true" if crag else "false")
    monkeypatch.setattr(cl, "_gather_local_items", lambda q, mode: (list(local_items), []))

    captured = {}

    def fake_run_agent(task, *, brief="", use_search=True, conversation="", on_event=None):
        captured.update(task=task, brief=brief, use_search=use_search)
        if on_event:
            on_event({"type": "status", "message": "agent working"})
        return types.SimpleNamespace(answer="ok", best_code="print(1)", best_output="1",
                                     success=True, tests_total=2, tests_passed=2)

    monkeypatch.setattr("backend.agent.loop.run_agent", fake_run_agent)
    events = list(cl.stream_chat_events(sid, CODE_Q))
    return events, captured, mem, sid


def _code_statuses(events):
    return " ".join(e.get("message", "") for e in events if e["type"] == "status").lower()


def test_code_from_paper_strong_uses_paper_as_spec(tmp_path, monkeypatch):
    # 3 strong relevant chunks -> STRONG and not thin: the paper alone is the spec (no GitHub).
    local = [_local(0.80), _local(0.70), _local(0.60)]
    events, cap, mem, sid = _drive_code(monkeypatch, tmp_path, local)

    assert cap["brief"], "the extracted algorithm spec should be passed as the agent brief"
    assert "local PDF passage" in cap["brief"]
    assert cap["use_search"] is False                # strong + enough chunks -> no GitHub supplement
    content = mem.get_turns(sid)[-1]["content"]
    assert "Implemented from your research library" in content and "MVDR Paper" in content
    assert "found the algorithm in your pdfs" in _code_statuses(events)


def test_code_from_paper_thin_supplements_with_github(tmp_path, monkeypatch):
    # one relevant chunk -> PARTIAL (and thin): still build a spec, but supplement with GitHub.
    local = [_local(0.80)]
    events, cap, mem, sid = _drive_code(monkeypatch, tmp_path, local)

    assert cap["brief"]                               # spec extracted from the single chunk
    assert cap["use_search"] is True                 # thin -> GitHub references fill the gaps
    assert "github references to fill gaps" in _code_statuses(events)


def test_code_not_in_papers_falls_back_to_github_only(tmp_path, monkeypatch):
    # nothing relevant -> NONE: no paper spec, GitHub-reference code path as before.
    local = [_local(0.20, "Irrelevant")]
    events, cap, mem, sid = _drive_code(monkeypatch, tmp_path, local)

    assert cap["brief"] == ""                         # no spec
    assert cap["use_search"] is True
    content = mem.get_turns(sid)[-1]["content"]
    assert "Implemented from your research library" not in content
    assert "not in your pdfs" in _code_statuses(events)


def test_code_crag_disabled_skips_paper_lookup(tmp_path, monkeypatch):
    # CRAG off: the paper lookup is skipped entirely; GitHub-reference code path.
    local = [_local(0.90)]                            # would be relevant, but never consulted
    events, cap, mem, sid = _drive_code(monkeypatch, tmp_path, local, crag=False)

    assert cap["brief"] == ""
    assert cap["use_search"] is True
    assert "checking your papers" not in _code_statuses(events)


# ======================================================================
# _gather_pass: deterministic multi-angle merge + the external timeout backstop
# ======================================================================
def test_gather_pass_times_out_sets_flag():
    tr = tracing.start_trace("test")

    def slow(q, k):
        time.sleep(0.3)                              # outlives the timeout below
        return ([{"source_type": "web", "title": "late", "text": "x", "url": "http://x/late"}], [])

    items, warnings, timed_out = cl._gather_pass(
        ["q"], slow, lambda i, q: 1, trace=tr, span_name="external_search", timeout=0.02)

    assert timed_out is True
    assert items == []                               # the slow result is dropped, not awaited


def test_gather_pass_multi_query_merges_in_query_order_with_correct_topk():
    tr = tracing.start_trace("test")
    calls = []

    def gather(q, k):
        calls.append((q, k))
        return ([{"source_type": "web", "title": q, "text": q, "url": "http://x/" + q}], [])

    items, warnings, timed_out = cl._gather_pass(
        ["q0", "q1", "q2"], gather, lambda i, q: 10 if i == 0 else 5,
        trace=tr, span_name="external_search")

    assert len(calls) == 3                           # one gather per angle
    by_q = dict(calls)
    assert by_q["q0"] == 10 and by_q["q1"] == 5 and by_q["q2"] == 5   # idx-aware top_k
    assert [it["title"] for it in items] == ["q0", "q1", "q2"]        # stable, query-order merge
    assert timed_out is False
