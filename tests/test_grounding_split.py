"""Grounding policy split: code generation uses the LLM's own knowledge (never refuses for
lack of sources, may use libraries), while the prose prompt still cites but no longer forbids
a code example just because the sources lack code. Runtime sandbox safety wording is kept."""
from backend.agent.loop import _GEN_SYSTEM
from webapp.chat_logic import SYSTEM_PROMPT


def test_gen_prompt_allows_own_knowledge_and_libraries():
    s = _GEN_SYSTEM.lower()
    assert "own expert knowledge" in s
    assert "never refuse" in s
    assert "third-party" in s                       # may use numpy/scipy/etc.
    assert "standard library only" not in s         # the old domain-locking rule is gone


def test_gen_prompt_keeps_runtime_safety():
    s = _GEN_SYSTEM.lower()
    assert "no network" in s and "no file access" in s and "input()" in s


def test_prose_prompt_no_longer_forces_code_from_sources():
    assert "Do NOT refuse because the sources lack code" in SYSTEM_PROMPT
    assert "read the method from the cited" not in SYSTEM_PROMPT   # old coupling removed
