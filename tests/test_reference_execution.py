"""Reference-execution testing: tests derive EXPECTED values by RUNNING a reference oracle, never
by guessing literals. The assembled harness is plain Python (the sandbox just runs it), so we exec
it in-process to prove the oracle decides pass/fail — no Docker, no LLM."""
import contextlib
import io
import re

import backend.agent.loop as loop

REF = "def add(a, b):\n    return a + b\n"
# A CORRECT candidate written differently than the reference (different expression, same result).
CAND_OK = "def add(a, b):\n    return b + a  # commuted on purpose\n"
CAND_BAD = "def add(a, b):\n    return a - b  # wrong\n"
# Tests get EXPECTED from the oracle `ref`, never from a hand-written literal.
TESTS_REF = (
    "def test_add_matches_reference():\n"
    "    for a, b in [(1, 2), (5, 7), (-3, 4), (0, 0)]:\n"
    "        assert add(a, b) == ref.add(a, b)\n"
)


def _run(script: str) -> str:
    buf = io.StringIO()
    # Capture stderr too: the runner prints tracebacks (e.g. NameError) to stderr.
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(compile(script, "<script>", "exec"), {})
    return buf.getvalue()


def _tally(out: str):
    m = re.search(r"TESTS_PASSED\s+(\d+)\s*/\s*(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def test_expected_is_computed_by_executing_the_reference_not_literals():
    script = loop._build_script(CAND_OK, TESTS_REF, loop._TEST_FOOTER, reference_src=REF)
    # The oracle is embedded + executed; the tests read expected from ref.*, not a guessed number.
    assert "ref = _types.ModuleType('ref')" in script and "_REF_SRC = " in script
    assert "ref.add(" in TESTS_REF and not re.search(r"==\s*\d", TESTS_REF)  # no literal expected
    out = _run(script)
    assert _tally(out) == (1, 1)              # a correctly-but-differently-written candidate matches


def test_reference_catches_a_wrong_candidate():
    out = _run(loop._build_script(CAND_BAD, TESTS_REF, loop._TEST_FOOTER, reference_src=REF))
    assert _tally(out)[0] == 0                # failure decided by the oracle, not a guessed value


def test_candidate_cannot_call_the_oracle_to_cheat():
    # The candidate runs in its own module whose globals do NOT include `ref`.
    cheat = "def add(a, b):\n    return ref.add(a, b)  # try to use the oracle\n"
    out = _run(loop._build_script(cheat, TESTS_REF, loop._TEST_FOOTER, reference_src=REF))
    assert _tally(out)[0] == 0 and "NameError" in out


def test_reference_and_candidate_get_the_SAME_random_inputs(monkeypatch):
    # Held-out style: random inputs generated ONCE and passed to BOTH sides; the seeded footer
    # seeds the RNG once for the whole run -> candidate and expected share inputs + seed.
    tests_rand = (
        "def test_hidden_add():\n"
        "    import random\n"
        "    for _ in range(25):\n"
        "        a, b = random.randint(-99, 99), random.randint(-99, 99)\n"
        "        assert add(a, b) == ref.add(a, b)\n"
    )
    script = loop._build_script(CAND_OK, tests_rand, loop._seeded_footer(1234), reference_src=REF)
    assert "random.seed(1234)" in script                   # one shared seed value for the whole run
    assert "_SOL_SRC = " in script and "_REF_SRC = " in script   # same process, both modules
    assert _tally(_run(script)) == (1, 1)


def test_no_oracle_falls_back_to_property_tests():
    # Without a reference, the harness has no `ref` block; property tests still run.
    prop = "def test_commutes():\n    assert add(2, 5) == add(5, 2)\n"
    script = loop._build_script(CAND_OK, prop, loop._TEST_FOOTER, reference_src="")
    assert "ModuleType('ref')" not in script
    assert _tally(_run(script)) == (1, 1)


class _Recorder:
    is_available = True

    def __init__(self):
        self.user = None

    def stream_chat(self, messages, system="", max_tokens=0, temperature=0):
        self.user = messages[-1]["content"] if messages else ""
        return ["def test_x():\n    assert True\n"]


def test_tests_prompt_instructs_oracle_use_only_when_reference_available():
    rec = _Recorder()
    loop._generate_tests(rec, "add two numbers", "- add(a,b)->a+b",
                         task_type="deterministic", use_reference=True)
    assert "REFERENCE ORACLE" in rec.user and "ref.fn" in rec.user
    rec2 = _Recorder()
    loop._generate_tests(rec2, "add two numbers", "- add(a,b)->a+b",
                         task_type="deterministic", use_reference=False)
    assert "REFERENCE ORACLE" not in rec2.user           # no oracle -> no oracle clause


def test_reference_tests_toggle():
    import os
    os.environ.pop("AGENT_REFERENCE_TESTS", None)
    assert loop.reference_tests_enabled() is True
    os.environ["AGENT_REFERENCE_TESTS"] = "false"
    try:
        assert loop.reference_tests_enabled() is False
    finally:
        os.environ.pop("AGENT_REFERENCE_TESTS", None)
