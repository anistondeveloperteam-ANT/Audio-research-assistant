"""Routing decisions: which queries go to the code agent vs the prose/citation pipeline."""
import pytest

from backend.answering.code_intent import is_code_intent

CODE = [
    "Give me RTF-MVDR python code",                              # the reported bug
    "Give me code to price a European option with Black-Scholes",
    "implement quicksort",
    "simulate a damped pendulum",
    "write a function that sorts a list",
    "python code for the FFT",
    "benchmark mergesort vs quicksort",
    "show me a script that scrapes a page",
    "generate an implementation of Dijkstra",
    "refactor this loop",
]

NOT_CODE = [
    "What is MVDR beamforming?",
    "Explain the Black-Scholes model",
    "How does transformer attention work?",
    "Compare Raft and Paxos",
    "Summarize the latest diffusion-model papers",
    "What does the function of the hippocampus involve?",   # 'function' must not trigger
    "Give me the latest papers on speech enhancement",
    "",
    "   ",
]


@pytest.mark.parametrize("q", CODE)
def test_code_intent_true(q):
    assert is_code_intent(q) is True, q


@pytest.mark.parametrize("q", NOT_CODE)
def test_code_intent_false(q):
    assert is_code_intent(q) is False, q
