"""Reference-code retrieval: topic extraction + REFERENCE-marked GitHub blocks (github_search
mocked, offline)."""
import types

import backend.agent.reference_code as rc


def test_topic_of_strips_request_boilerplate():
    assert rc.topic_of("Give me RTF-MVDR python code") == "RTF-MVDR"
    t = rc.topic_of("Give me code to price a European option with Black-Scholes").lower()
    assert "black-scholes" in t
    assert "give" not in t and "code" not in t and "python" not in t


def test_fetch_reference_marks_source_and_license(monkeypatch):
    fake = types.SimpleNamespace(title="owner/mvdr — README (900★)", url="http://x",
                                 text="def mvdr(cov, steering):\n    ...", snippet="", license="MIT")
    monkeypatch.setattr("backend.external_search.github_search.github_search",
                        lambda q, max_repos=2: [fake])
    out = rc.fetch_reference_code("RTF-MVDR python code")
    assert "REFERENCE" in out and "MIT" in out
    assert "do NOT copy" in out and "def mvdr" in out


def test_fetch_reference_empty_on_failure(monkeypatch):
    def boom(q, max_repos=2):
        raise RuntimeError("network down")

    monkeypatch.setattr("backend.external_search.github_search.github_search", boom)
    assert rc.fetch_reference_code("anything") == ""


def test_fetch_reference_empty_when_no_results(monkeypatch):
    monkeypatch.setattr("backend.external_search.github_search.github_search",
                        lambda q, max_repos=2: [])
    assert rc.fetch_reference_code("obscure topic xyz") == ""
