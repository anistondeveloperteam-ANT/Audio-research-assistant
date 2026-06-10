"""Tests for the LangGraph-orchestrated research pipeline (steps mocked — no network)."""
import pytest

import backend.agent.research_agent as ra
import backend.agent.langgraph_research as lg

langgraph_installed = lg.langgraph_available()


class _FakeProvider:
    is_available = True

    def unavailable_message(self):
        return "n/a"


def _mock_steps(monkeypatch, reflect_done_after=1):
    """Stub the LLM/search steps so the graph runs offline and deterministically."""
    monkeypatch.setattr(ra, "_plan", lambda p, q: ["sub one", "sub two"])

    state = {"searches": 0, "reflects": 0}

    def fake_search(query, k):
        state["searches"] += 1
        return [{"url": f"http://ex/{state['searches']}", "title": query, "text": "evidence " + query}], []

    monkeypatch.setattr(ra, "_search", fake_search)
    monkeypatch.setattr(ra, "_synthesize", lambda p, q, subs, ev, prog: "finding: " + ", ".join(subs))

    def fake_reflect(p, q, findings):
        state["reflects"] += 1
        done = state["reflects"] >= reflect_done_after
        return {"done": done, "next_queries": [] if done else ["follow up query"]}

    monkeypatch.setattr(ra, "_reflect", fake_reflect)
    monkeypatch.setattr(ra, "_write_report", lambda p, q, ev, f: f"REPORT[{q}] notes={len(f)}")
    monkeypatch.setattr(lg, "get_provider", lambda *a, **k: _FakeProvider())
    return state


@pytest.mark.skipif(not langgraph_installed, reason="langgraph not installed")
def test_graph_runs_plan_search_synth_report(monkeypatch):
    _mock_steps(monkeypatch, reflect_done_after=1)
    report = lg.run("How does MVDR work?", max_rounds=3)
    assert report.report == "REPORT[How does MVDR work?] notes=1"
    assert report.sub_questions == ["sub one", "sub two"]
    assert report.rounds == 1
    assert len(report.sources) == 2          # two sub-questions, one item each


@pytest.mark.skipif(not langgraph_installed, reason="langgraph not installed")
def test_graph_loops_back_to_search_until_done(monkeypatch):
    state = _mock_steps(monkeypatch, reflect_done_after=2)   # wants a 2nd round
    report = lg.run("compare A and B", max_rounds=3)
    assert report.rounds == 2
    assert state["searches"] == 3            # round1: 2 sub-qs + round2: 1 follow-up
    assert "REPORT[compare A and B]" in report.report


@pytest.mark.skipif(not langgraph_installed, reason="langgraph not installed")
def test_round_budget_caps_the_loop(monkeypatch):
    state = _mock_steps(monkeypatch, reflect_done_after=99)   # never satisfied
    report = lg.run("endless question", max_rounds=2)
    assert report.rounds == 2                # capped by max_rounds, not by the model
    assert state["reflects"] == 1            # reflect skipped on the final (capped) round


def test_empty_question_returns_error():
    assert lg.run("   ").error


def test_falls_back_when_langgraph_missing(monkeypatch):
    monkeypatch.setattr(lg, "langgraph_available", lambda: False)
    called = {}

    def fake_research(q, **k):
        called["q"] = q
        return ra.ResearchReport(question=q, report="builtin")

    monkeypatch.setattr(ra, "research", fake_research)
    out = lg.run("fallback please")
    assert out.report == "builtin" and called["q"] == "fallback please"
