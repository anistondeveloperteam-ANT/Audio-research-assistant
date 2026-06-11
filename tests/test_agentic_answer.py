"""Offline tests for the web-chat agentic answer helpers."""

from backend.answering import agentic_answer as aa


def test_parse_json_object_handles_embedded_json():
    out = aa.parse_json_object('extra {"ok": true, "score": 91} tail')
    assert out["ok"] is True
    assert out["score"] == 91


def test_extract_python_blocks_prefers_longest_python_fence():
    answer = (
        "```python\nprint('short')\n```\n"
        "```python\nimport math\nprint(math.sqrt(16))\n```\n"
    )
    blocks = aa.extract_python_blocks(answer)
    assert blocks[0].startswith("import math")
    assert "sqrt" in blocks[0]


def test_followup_query_uses_verifier_query_first():
    verdict = {"followup_query": "mvdr beamforming pesq evaluation"}
    assert aa.followup_query("question", verdict) == "mvdr beamforming pesq evaluation"


def test_followup_query_falls_back_to_missing_evidence():
    verdict = {"missing_evidence": ["dataset name", "metric values"]}
    query = aa.followup_query("How was it evaluated?", verdict)
    assert "How was it evaluated?" in query
    assert "dataset name" in query


def test_verification_footer_reports_score_and_run():
    footer = aa.verification_footer(
        verdict={"ok": True, "score": 90},
        rounds=2,
        run_info={"summary": "OK in 0.2s"},
    )
    assert "90/100" in footer
    assert "2 round" in footer
    assert "OK in 0.2s" in footer


def test_max_verify_rounds_respects_env_and_caps(monkeypatch):
    monkeypatch.setenv("AGENTIC_MAX_VERIFY_ROUNDS", "3")
    assert aa.max_verify_rounds() == 3
    monkeypatch.setenv("AGENTIC_MAX_VERIFY_ROUNDS", "99")
    assert aa.max_verify_rounds() == 5        # hard ceiling
    monkeypatch.setenv("AGENTIC_MAX_VERIFY_ROUNDS", "0")
    assert aa.max_verify_rounds() == 1        # floor
    monkeypatch.setenv("AGENTIC_MAX_VERIFY_ROUNDS", "notanint")
    assert aa.max_verify_rounds() == 3        # bad value -> default


def test_verification_passed_requires_ok_and_threshold(monkeypatch):
    monkeypatch.setenv("AGENTIC_MIN_VERIFY_SCORE", "80")
    assert aa.verification_passed({"ok": True, "score": 85}) is True
    assert aa.verification_passed({"ok": True, "score": 40}) is False   # below bar -> not verified
    assert aa.verification_passed({"ok": False, "score": 99}) is False


def test_run_best_python_block_skips_when_no_code():
    assert aa.run_best_python_block("No runnable code here.") is None
