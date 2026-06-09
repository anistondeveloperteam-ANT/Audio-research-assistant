"""Tests for the autonomous research agent. Mocked LLM + search; no network/key."""
import backend.agent.research_agent as ra
import backend.answering.reviewer as reviewer_mod


class _FakeProvider:
    is_available = True
    name = "openai"
    model = "test"

    def unavailable_message(self):
        return "LLM not available"

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0):
        s = system.lower()
        if "research planner" in s:
            return ['{"subquestions": ["what is X", "how does X compare"]}']
        if "research analyst" in s:
            return ["- X is a method [1]\n- It compares well [2]"]
        if "research critic" in s:
            return ['{"done": true, "gaps": [], "next_queries": []}']
        if "research writer" in s:
            return ["# Report\nDirect answer [1].\n\n## Key takeaways\n- It works [2]"]
        return ['{"recommendation": "accept", "summary": "ok", "strengths": [], '
                '"weaknesses": [], "suggestions": []}']


class _Src:
    def __init__(self, title, url, text, st="web"):
        self._d = {"source_type": st, "title": title, "url": url}
        self.text = text

    def to_public(self):
        return dict(self._d)


def _fake_search(query, max_results=8):
    return [
        _Src("Paper A", "http://a.com", "A explains X in depth."),
        _Src("Repo B", "http://b.com", "B implements X."),
    ], []


def _patch(monkeypatch):
    monkeypatch.setattr(ra, "get_provider", lambda: _FakeProvider())
    monkeypatch.setattr(ra, "gather_external_evidence", _fake_search)
    # The reviewer uses its own get_provider(); patch it for the self-review path.
    monkeypatch.setattr(reviewer_mod, "get_provider", lambda: _FakeProvider())


def test_research_produces_cited_report(monkeypatch):
    _patch(monkeypatch)
    res = ra.research("How does X work?", max_rounds=2, per_query_results=4, do_review=False)
    assert not res.error
    assert "Report" in res.report
    assert res.sub_questions == ["what is X", "how does X compare"]
    assert len(res.sources) >= 1
    assert res.rounds >= 1


def test_research_stops_when_reflection_done(monkeypatch):
    _patch(monkeypatch)
    events = []
    res = ra.research("Q?", max_rounds=5, do_review=False, on_event=events.append)
    # The critic returns done=true, so it should stop after round 1.
    assert res.rounds == 1
    assert any(e["type"] == "done" for e in events)


def test_research_self_review_path(monkeypatch):
    _patch(monkeypatch)
    res = ra.research("Q?", max_rounds=1, do_review=True)
    assert res.review is not None
    assert res.review.get("recommendation") == "accept"


def test_research_empty_question():
    res = ra.research("   ")
    assert res.error


def test_research_provider_unavailable(monkeypatch):
    class _NA:
        is_available = False

        def unavailable_message(self):
            return "set OPENAI_API_KEY"
    monkeypatch.setattr(ra, "get_provider", lambda: _NA())
    res = ra.research("Q?")
    assert "OPENAI_API_KEY" in res.error


def test_dedup_and_evidence_numbering():
    items = [
        {"title": "A", "url": "http://a.com/", "text": "x", "source_type": "web"},
        {"title": "A", "url": "http://a.com", "text": "x", "source_type": "web"},
    ]
    # Same URL (trailing slash) -> one key.
    assert ra._item_key(items[0]) == ra._item_key(items[1])
    ev = ra._format_evidence(items)
    assert ev.startswith("[1] (web) A")
