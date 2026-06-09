"""Tests for the automated peer reviewer — mocked LLM, no network/key."""
import json

from backend.answering import reviewer


class _Fake:
    name = "openai"
    model = "t"
    is_available = True

    def __init__(self, resp):
        self._r = resp

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0):
        return [self._r]


def test_review_parses_structured_output(monkeypatch):
    resp = json.dumps({
        "summary": "S", "strengths": ["clear method"], "weaknesses": ["small N"],
        "questions": [], "suggestions": ["add a baseline"],
        "scores": {"novelty": 7, "soundness": 6, "clarity": 8, "significance": 7},
        "recommendation": "minor revision", "confidence": 4,
    })
    monkeypatch.setattr(reviewer, "get_provider", lambda: _Fake(resp))
    r = reviewer.review("some paper text")
    assert r["recommendation"] == "minor revision"
    assert r["scores"]["clarity"] == 8
    assert "clear method" in r["strengths"]


def test_review_handles_json_wrapped_in_prose(monkeypatch):
    monkeypatch.setattr(reviewer, "get_provider",
                        lambda: _Fake('Here is the review: {"recommendation": "reject"} done'))
    assert reviewer.review("x")["recommendation"] == "reject"


def test_review_empty_input():
    assert reviewer.review("") == {}


def test_review_llm_unavailable(monkeypatch):
    class _NA:
        is_available = False
    monkeypatch.setattr(reviewer, "get_provider", lambda: _NA())
    assert "error" in reviewer.review("text")


def test_parse_json_helper():
    assert reviewer._parse_json('{"a": 1}') == {"a": 1}
    assert reviewer._parse_json("garbage") == {}
