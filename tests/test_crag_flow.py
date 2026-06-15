"""CRAG retrieval flow in webapp/chat_logic.py: grade local PDF evidence, then act.

Fully offline — local/external retrieval and the deep-query planner are mocked, and we stop
consuming the event stream at the `sources` event (the grade + external decision is complete by
then, before any LLM generation)."""
import webapp.chat_logic as cl
from backend.memory.store import MemoryStore

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
# CRAG disabled -> original concurrent sweep still runs (local + external together)
# ----------------------------------------------------------------------
def test_crag_disabled_uses_legacy_concurrent_sweep(tmp_path, monkeypatch):
    local = [_local(0.80), _local(0.72)]             # would be STRONG, but CRAG is off
    events, external_calls, sources = _drive(monkeypatch, tmp_path, local, web=True, crag=False)

    assert external_calls, "legacy sweep always runs external when web search is on"
    titles = [s["title"] for s in sources]
    assert "MVDR Paper" in titles and "WebResult" in titles
