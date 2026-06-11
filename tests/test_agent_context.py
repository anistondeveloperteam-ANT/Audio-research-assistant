"""
A1/A2: the code agent must stay on-topic for follow-ups like "give me code for this".
These test the brief-building (A1) and the relevance gate (A2) — no LLM/Docker needed.
"""
from backend.agent.loop import _build_brief


def test_build_brief_includes_conversation_topic():
    conv = ("User: Explain MVDR beamforming for an 8-channel mic array.\n"
            "Assistant: MVDR minimizes output power subject to a distortionless constraint...")
    brief = _build_brief("give me code for this", "", "", conv)
    assert "MVDR" in brief and "beamforming" in brief          # prior topic carried in
    assert brief.index("MVDR") < brief.index("give me code for this")  # topic framed first


def test_build_brief_without_conversation_is_just_goal():
    brief = _build_brief("sort a list of integers", "", "")
    assert "# Goal" in brief and "sort a list of integers" in brief
    assert "Conversation so far" not in brief


def test_is_relevant_flags_offtopic():
    from backend.answering.reviewer import is_relevant
    assert is_relevant({"scores": {"relevance": 2}, "recommendation": "reject"}) is False
    assert is_relevant({"recommendation": "reject"}) is False
    assert is_relevant({"scores": {"relevance": 9}, "recommendation": "accept"}) is True
    assert is_relevant({}) is True            # no signal -> never block


def test_offtopic_attempt_never_wins_score():
    from backend.agent.loop import _score, Attempt
    from backend.agent.code_runner import RunResult
    ok = RunResult(True, 0, "out", "", 0.1)
    on_topic = Attempt(1, "code", ok, {"relevant": True, "score": 50})
    off_topic = Attempt(2, "code", ok, {"relevant": False, "score": 99})
    assert _score(on_topic) > _score(off_topic)
