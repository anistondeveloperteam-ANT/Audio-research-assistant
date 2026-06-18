"""Requested code-agent outputs stay visible even when the program prints large intermediate data.

Root cause fixed here: stdout was HEAD-truncated, so a requested value printed AFTER a big
array/matrix/dataset dump was silently dropped from both the user-facing output and the completeness
gate. The fix keeps BOTH ends of stdout (head + larger tail) and instructs the demo driver not to
flood stdout and to print the requested values in a final labelled block. Fully offline — the
sandbox run is mocked; no Docker or LLM needed.
"""
import pytest

import backend.agent.loop as loop
from backend.agent.code_runner import OUTPUT_CAP, RunResult, _cap, clip_keep_ends
from backend.agent.loop import DEMO_OUTPUT_CAP, _apply_output_gates, _capture_and_check, _check_completeness


# ======================================================================
# clip_keep_ends / _cap — keep the TAIL (where the requested result lives)
# ======================================================================
def test_clip_keep_ends_preserves_the_final_block():
    big = "X" * 50_000
    clipped = clip_keep_ends(big + "\nREQUESTED RESULT: answer=42\n", 12_000)
    assert "REQUESTED RESULT: answer=42" in clipped         # the LAST line survives
    assert "elided" in clipped                               # truncation marker kept
    assert len(clipped) <= 12_000 + 80                       # bounded


def test_clip_keep_ends_keeps_both_ends():
    clipped = clip_keep_ends("HEAD-MARKER" + "Y" * 50_000 + "END-MARKER", 12_000)
    assert clipped.startswith("HEAD-MARKER")                 # early context kept too
    assert clipped.endswith("END-MARKER")                    # final block kept


def test_clip_keep_ends_noop_when_within_limit():
    assert clip_keep_ends("hello", 12_000) == "hello"
    assert clip_keep_ends("", 12_000) == ""


def test_cap_is_tail_preserving_not_head_only():
    capped = _cap("A" * (OUTPUT_CAP + 5_000) + "TAIL-VALUE")
    assert "TAIL-VALUE" in capped                            # no longer dropped by head-only cut
    assert len(capped) <= OUTPUT_CAP + 80


# ======================================================================
# Completeness is measured on the ACTUAL (clipped) stdout, not print presence
# ======================================================================
def test_completeness_present_when_value_in_final_block():
    stdout = "intermediate noise...\n=== RESULTS ===\nkinetic energy (J): 5.0\n"
    assert _check_completeness(["kinetic energy"], stdout) == []


def test_completeness_missing_when_value_buried_in_elided_middle():
    # A value PRINTED but flooded into the elided middle is NOT in captured stdout -> missing.
    head = "0.1, " * 1_500            # fills the head window
    tail = "9.9, " * 2_600            # fills the tail window
    clipped = clip_keep_ends(head + "\nburied metric: 9.9\n" + tail, DEMO_OUTPUT_CAP)
    assert "buried metric" not in clipped
    assert _check_completeness(["buried metric"], clipped) == ["buried metric"]


def test_completeness_missing_when_label_absent_from_stdout():
    # A print STATEMENT in the code is irrelevant: the gate checks the captured stdout.
    assert _check_completeness(["energy"], "only the period: 2.0\n") == ["energy"]


# ======================================================================
# Prompts / gate directive carry the no-flood + final-block rules
# ======================================================================
def test_driver_prompt_has_antiflood_and_final_block_rules():
    p = loop._DRIVER_SYSTEM.lower()
    assert "compact summary" in p
    assert "never print a whole large array" in p
    assert "final" in p and "last" in p
    assert "print every value the request asks for" in p     # existing label rule NOT regressed


def test_gate_refine_directive_has_antiflood_rule():
    v = _apply_output_gates({"verified": True}, wants_output=True, output="x", missing=["mean"])
    fb = (v.get("feedback") or "").lower()
    assert "final" in fb and "summary" in fb
    assert v["verified"] is False and v["done"] is False     # gate still downgrades honestly


def test_gate_never_resurrects_a_failed_verdict():
    v = _apply_output_gates({"verified": False}, wants_output=True, output="", missing=["x"])
    assert v == {"verified": False}                          # untouched (no-resurrection guard kept)


# ======================================================================
# 3-domain demo: a huge intermediate dump THEN the requested value -> value stays visible
# ======================================================================
_HUGE = "[" + ", ".join(str(i % 10) for i in range(20_000)) + "]\n"   # >40k chars of "array"


@pytest.mark.parametrize("domain,deliverable,final_line,value", [
    ("fft peak frequency",   "peak frequency", "peak frequency (Hz): 5.0", "5.0"),
    ("matrix determinant",   "determinant",    "determinant: 42.0",        "42.0"),
    ("dataset mean",         "mean",           "dataset mean: 3.14",       "3.14"),
])
def test_capture_and_check_surfaces_requested_value_after_large_dump(
        monkeypatch, domain, deliverable, final_line, value):
    simulated = _HUGE + "=== RESULTS ===\n" + final_line + "\n"        # dump first, answer last
    monkeypatch.setattr(loop, "_generate_demo_driver", lambda *a, **k: "print('demo')")
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: RunResult(True, 0, simulated, "", 0.1))

    output, missing = _capture_and_check(None, domain, "requirements", "code", [deliverable])

    assert value in output and final_line in output          # requested value survived the clip
    assert missing == []                                     # completeness passes — value visible
    assert len(output) <= DEMO_OUTPUT_CAP + 80               # still bounded


def test_capture_and_check_flags_value_flooded_out_of_both_ends(monkeypatch):
    # The requested value is sandwiched between two huge dumps -> falls in the elided middle ->
    # not in captured stdout -> completeness FAILS (a buried print is treated as missing).
    big = "9, " * 6_000
    simulated = big + "\nsecret metric: 7.0\n" + big
    monkeypatch.setattr(loop, "_generate_demo_driver", lambda *a, **k: "print('demo')")
    monkeypatch.setattr(loop, "run_python_auto",
                        lambda code, **k: RunResult(True, 0, simulated, "", 0.1))

    output, missing = _capture_and_check(None, "task", "requirements", "code", ["secret metric"])

    assert "secret metric" not in output
    assert missing == ["secret metric"]
