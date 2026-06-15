"""
The coding agent's test-first generate -> run-vs-tests -> refine loop (AlphaCodium, 2401.08500).

    REQUIREMENTS : the LLM restates the task as a concrete requirements checklist
                   (using the conversation topic), so the code addresses THIS request.
    TESTS        : the LLM writes 5-8 concrete correctness tests for the task (the
                   acceptance criteria), derived from the algorithm itself.
    SOLUTION     : the LLM writes modular code; for each round we try up to two
                   candidates (best-of-2) and keep the one that passes more tests.
    RUN-VS-TESTS : solution + tests run together in a throwaway Docker sandbox; a
                   candidate is accepted ONLY when ALL generated tests pass (never on
                   "it ran without error") AND it implements the requested algorithm
                   (relevance gate). If two rounds fail, escalate to AGENT_MODEL_STRONG.

It keeps the best attempt and, if tests never fully pass, returns it honestly labelled
"partially verified — N/M tests passing". Optionally it first fetches 1-2 stars-first
GitHub reference implementations of the named algorithm to ADAPT (never copy).

The THINK -> EXECUTE -> REFLECT control-loop skeleton and the constant-size memory idea
were adapted (original code) from `auto-deep-researcher-24x7` (Apache-2.0); the test-first
generation/acceptance is adapted from AlphaCodium. A mid-flight HUMAN_DIRECTIVE file can
still steer the loop between rounds.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.agent.anticheat import anticheat_enabled, scan_for_cheating
from backend.agent.code_runner import RunResult, docker_available, run_python_auto
from backend.agent.hooks import pre_run
from backend.llm.streaming_provider import get_provider
from backend.observability import tracing  # no-op unless LANGFUSE_ENABLED=true

# Budgets are generous because reasoning models (GPT-5 / o-series) spend tokens
# "thinking" before emitting the code/JSON.
MAX_ITERS = int(os.getenv("AGENT_MAX_ITERS", "4"))
GEN_MAX_TOKENS = int(os.getenv("AGENT_GEN_MAX_TOKENS", "5000"))
REFLECT_MAX_TOKENS = int(os.getenv("AGENT_REFLECT_MAX_TOKENS", "2000"))
Event = Dict[str, Any]
OnEvent = Callable[[Event], None]


def hidden_tests_enabled() -> bool:
    """Live read (AGENT_HIDDEN_TESTS, default on): run held-out hidden tests + invariants on
    random inputs at final acceptance, so passing the visible tests is never enough."""
    return os.getenv("AGENT_HIDDEN_TESTS", "true").strip().lower() not in ("0", "false", "no", "off")


def verify_seeds() -> int:
    """Live read (AGENT_VERIFY_SEEDS, default 3): independent random seeds the held-out suite must
    pass on — passing on some but not all seeds is a fluke, not a verified solution."""
    try:
        return max(1, int(os.getenv("AGENT_VERIFY_SEEDS", "3")))
    except (TypeError, ValueError):
        return 3


def parallel_n() -> int:
    """Live read (AGENT_PARALLEL_N, default 4, clamped 1-8): how many candidate solutions to
    generate + run CONCURRENTLY each round (best-of-N). 1 = serial. The sandbox semaphore
    (AGENT_MAX_CONCURRENT_SANDBOXES) bounds how many containers actually run at once."""
    try:
        return max(1, min(8, int(os.getenv("AGENT_PARALLEL_N", "4"))))
    except (TypeError, ValueError):
        return 4


def reference_tests_enabled() -> bool:
    """Live read (AGENT_REFERENCE_TESTS, default on): derive each test's EXPECTED value by RUNNING
    a reference oracle in the sandbox, instead of letting the test-LLM guess literal outputs. Off
    falls back to property/legacy tests."""
    return os.getenv("AGENT_REFERENCE_TESTS", "true").strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Attempt:
    iteration: int
    code: str
    result: RunResult
    verdict: Dict[str, Any]


@dataclass
class AgentResult:
    task: str
    success: bool
    best_code: str
    best_output: str
    answer: str
    attempts: List[Attempt] = field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    verification: str = "failed"      # verified | partial | rejected_cheating | failed
    hidden_passed: int = 0
    hidden_total: int = 0
    cheat_flags: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------
def _complete(provider, system: str, user: str, max_tokens: int, temperature: float = 0.2) -> str:
    return "".join(provider.stream_chat(
        [{"role": "user", "content": user}], system=system,
        max_tokens=max_tokens, temperature=temperature,
    )).strip()


def _extract_code(text: str) -> str:
    """Pull the Python source out of an LLM reply (handles ``` fences or raw code)."""
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


_GEN_SYSTEM = (
    "You are an expert software and algorithms engineer. Implement the requested algorithm or "
    "task in Python using your OWN expert knowledge of how it works. NEVER refuse or apologize "
    "for a lack of reference material or sources — your correctness is judged by RUNNING the "
    "code, not by citations. "
    "You MAY use well-known third-party libraries when they are the right tool (e.g. numpy, "
    "scipy, pandas); the sandbox installs the packages you import, so import what you need. "
    "Write complete, self-contained, MODULAR code: small named functions for the core logic, "
    "each with a clear signature. A separate harness imports and exercises your functions against "
    "unit tests, so do NOT add a __main__ block, prints, or your own test runner — just define the "
    "functions. At RUNTIME the sandbox has no network, no file access, and no input() — do not use "
    "them (third-party imports are fine). The code must run to completion in a few seconds. "
    "Output ONLY the Python code — no explanation, no markdown."
)

_REQ_SYSTEM = (
    "You are a senior engineer. Restate the user's coding task as a short, concrete checklist of "
    "requirements (3-7 bullets): the function(s) to implement WITH their signatures, the inputs "
    "and outputs, and the key correctness properties to satisfy. Use the conversation context if "
    "given. Output ONLY the bullet list."
)

_REFERENCE_SYSTEM = (
    "You write a CLEAR, CORRECT REFERENCE implementation that serves as the trusted ORACLE for "
    "testing: its EXECUTED outputs are the expected values a separate candidate solution is "
    "compared against. Correctness matters far more than speed or elegance — use the simplest "
    "approach you are SURE is right. Implement the SAME function name(s) and signature(s) the task "
    "/ requirements specify, so the candidate can be called identically. You MAY use numpy / scipy "
    "/ stdlib. Define ONLY the functions — no __main__, no prints, no tests. At runtime there is no "
    "network, no file access, no input(). Output ONLY the Python code — no explanation, no markdown."
)

_TESTS_SYSTEM = (
    "You write rigorous but ROBUST unit tests as plain Python (no pytest, no unittest). Given a "
    "task and its requirements, write 5-7 focused test functions named test_<name>() that:\n"
    "- call the SOLUTION's functions directly (they are defined in the test scope);\n"
    "- use SELF-CONSISTENT inputs: the SAME calling convention and array shapes in EVERY test, "
    "matching ONE function signature — do NOT require the function to accept several input shapes;\n"
    "- compare floats with tolerances (math.isclose, or numpy.allclose with explicit rtol/atol), "
    "NEVER exact ==; build small CONCRETE inputs (use numpy if it helps);\n"
    "- obtain every EXPECTED value by COMPUTING it — from the reference oracle when one is provided "
    "(see below), otherwise as a PROPERTY that holds for any correct implementation. NEVER "
    "hand-write a literal expected number/output you imagined; a guessed expected value is the #1 "
    "cause of false failures;\n"
    "- each test must `assert` and raise AssertionError on failure.\n"
    "Do NOT define the solution, the reference, or any test runner. Output ONLY the test functions."
)

# Appended to the test/held-out USER prompt when a reference oracle is available. Tests derive
# `expected` by calling `ref.*` at runtime — so expected is EXECUTED, never guessed, and the
# candidate and the expected value share identical inputs/libraries by construction.
_ORACLE_CLAUSE = (
    "\n\nA correct REFERENCE ORACLE is available in the test scope as `ref`, exposing the SAME "
    "functions as the solution. To get an EXPECTED value, CALL THE REFERENCE on the SAME inputs you "
    "pass the solution — e.g. `expected = ref.fn(x); assert math.isclose(fn(x), expected, "
    "rel_tol=1e-6, abs_tol=1e-9)` (use numpy.allclose for arrays, == for exact ints/strings/"
    "containers). NEVER write an expected literal you imagined — the expected value MUST come from "
    "`ref`. Do NOT call `ref` inside the solution; only the tests use it."
)

_HIDDEN_SYSTEM = (
    "You write HELD-OUT acceptance tests that the implementer never sees. Given a task and its "
    "requirements, write 4-6 functions named test_hidden_<name>() that exercise the SAME required "
    "behavior as ordinary tests but on DIFFERENT, FRESH inputs — generate them with random / "
    "numpy.random (do NOT call seed yourself; the harness seeds globally) so each run uses new "
    "data, and do NOT reuse any example values. Get every EXPECTED value by COMPUTING it (the "
    "reference oracle when provided, else a property) — NEVER a guessed literal. Compare with "
    "tolerances (math.isclose / numpy.allclose), call the SOLUTION's functions directly, and "
    "`assert`. Their purpose is to catch code that special-cased or hardcoded the visible examples. "
    "Do NOT define the solution or a runner. Output ONLY the Python test functions."
)

_INVARIANTS_SYSTEM = (
    "You write INVARIANT checks: 1-3 functions named test_invariant_<name>() that assert "
    "mathematical PROPERTIES which must hold for ANY correct implementation, on RANDOM inputs "
    "(use random / numpy.random; do NOT seed — the harness seeds globally). Examples: a "
    "beamformer's distortionless constraint w^H d ~= 1 and output noise power <= input; "
    "Black-Scholes put-call parity C - P ~= S - K*exp(-rT), price >= 0, price monotonic in "
    "volatility. Use tolerances, call the SOLUTION's functions directly, and `assert`. Do NOT "
    "define the solution or a runner. Output ONLY the Python test functions."
)

# Task-type-specific guidance appended to the test/invariant generation USER prompts (the system
# prompts above stay fixed) so verification matches the KIND of task. The model DERIVES 2-4
# concrete properties appropriate to the type — a single "expected output" is meaningless for a
# stochastic simulation, so those are checked by invariants instead.
_TASK_TYPE_GUIDANCE = {
    "deterministic": (
        "\n\nTASK TYPE = DETERMINISTIC: a single correct output exists. Assert EXACT expected "
        "outputs on small concrete inputs (use math.isclose / numpy.allclose only for floats), "
        "PLUS 2-4 general properties any correct solution must satisfy (e.g. output length and "
        "ordering, idempotence, boundary/empty cases)."
    ),
    "simulation": (
        "\n\nTASK TYPE = SIMULATION / STOCHASTIC: there is NO single fixed output — do NOT assert "
        "one magic number. DERIVE 2-4 INVARIANTS / PROPERTIES on the REAL output: correct output "
        "shape/type; conservation laws (energy / mass / probability sums); values within physical "
        "or range bounds; expected convergence or a monotonic trend (e.g. a damped system's "
        "amplitude decreases over time); and seeded REPRODUCIBILITY (same seed -> identical "
        "output). Use tolerances generous enough for discretization/noise."
    ),
    "numeric_algorithm": (
        "\n\nTASK TYPE = NUMERIC ALGORITHM: assert DOMAIN INVARIANTS, not one value. DERIVE 2-4 "
        "mathematical properties that hold for ANY correct implementation, e.g. a beamformer's "
        "distortionless constraint w^H d ~= 1 and output noise power <= input; FFT / Parseval "
        "energy conservation; Black-Scholes put-call parity C - P ~= S - K*exp(-rT), price >= 0 "
        "and monotonic in volatility. Compare with tolerances."
    ),
}


def _task_type_hint(task_type: str) -> str:
    """Verification guidance for a task_type (deterministic | simulation | numeric_algorithm)."""
    return _TASK_TYPE_GUIDANCE.get((task_type or "").strip().lower(), "")


# Appended after (solution + generated tests): runs every test_* and prints a parseable tally.
_TEST_FOOTER = (
    "\n\n# === auto-appended test runner ===\n"
    "def __run_all_tests():\n"
    "    import traceback\n"
    "    g = dict(globals())\n"
    "    names = sorted(n for n, v in g.items() if n.startswith('test_') and callable(v))\n"
    "    passed = 0\n"
    "    for _n in names:\n"
    "        try:\n"
    "            g[_n]()\n"
    "            print('TEST', _n, 'PASS')\n"
    "            passed += 1\n"
    "        except Exception:\n"
    "            print('TEST', _n, 'FAIL')\n"
    "            traceback.print_exc()\n"
    "    print('TESTS_PASSED %d/%d' % (passed, len(names)))\n"
    "__run_all_tests()\n"
)


def _seeded_footer(seed: int) -> str:
    """Held-out runner that seeds the RNGs first, so each seed exercises different random inputs
    while staying reproducible. Same TESTS_PASSED k/n contract as _TEST_FOOTER."""
    return (
        "\n\n# === auto-appended held-out runner (seeded) ===\n"
        "def __run_all_tests():\n"
        "    import traceback, random\n"
        f"    random.seed({seed})\n"
        "    try:\n"
        f"        import numpy as _np; _np.random.seed({seed})\n"
        "    except Exception:\n"
        "        pass\n"
        "    g = dict(globals())\n"
        "    names = sorted(n for n, v in g.items() if n.startswith('test_') and callable(v))\n"
        "    passed = 0\n"
        "    for _n in names:\n"
        "        try:\n"
        "            g[_n]()\n"
        "            print('TEST', _n, 'PASS')\n"
        "            passed += 1\n"
        "        except Exception:\n"
        "            print('TEST', _n, 'FAIL')\n"
        "            traceback.print_exc()\n"
        "    print('TESTS_PASSED %d/%d' % (passed, len(names)))\n"
        "__run_all_tests()\n"
    )


def _restate_requirements(provider, task: str, conversation: str, reference: str) -> str:
    """(a) Restate the task as a concrete requirements checklist, using conversation context."""
    parts = []
    if conversation.strip():
        parts.append(f"CONVERSATION (the topic the code must address):\n{conversation}\n")
    parts.append(f"TASK:\n{task}")
    if reference:
        parts.append(f"\nREFERENCE (context only):\n{reference[:1500]}")
    parts.append("\nWrite the requirements checklist now.")
    out = _complete(provider, _REQ_SYSTEM, "\n".join(parts), REFLECT_MAX_TOKENS)
    return out or f"- Implement the task: {task}"


def _generate_reference(provider, task: str, requirements: str, task_type: str = "") -> str:
    """(oracle) A clear, correct reference implementation whose EXECUTED outputs become the expected
    values the tests assert against — so 'expected' is computed, never guessed. Returns '' on any
    failure, and the caller falls back to property/legacy tests."""
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\n"
            "Write the reference implementation now — the SAME functions the task requires.")
    try:
        return _extract_code(_complete(provider, _REFERENCE_SYSTEM, user, GEN_MAX_TOKENS))
    except Exception:
        return ""


def _generate_tests(provider, task: str, requirements: str, task_type: str = "",
                    use_reference: bool = False) -> str:
    """(b) Generate 5-8 concrete test_* functions that target THIS task (derived, not hardcoded),
    shaped by the task_type. When `use_reference`, expected values are computed by calling the
    reference oracle `ref.*` at runtime instead of being written as literals."""
    user = f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}" + _task_type_hint(task_type)
    if use_reference:
        user += _ORACLE_CLAUSE
    user += "\n\nWrite the test_* functions now (they call the solution's functions directly)."
    return _extract_code(_complete(provider, _TESTS_SYSTEM, user, GEN_MAX_TOKENS))


def _generate_solution(provider, task: str, requirements: str, tests: str, reference: str,
                       last_code: str, feedback: str, memory_summary: str = "",
                       temperature: float = 0.2, variant: str = "") -> str:
    """(c) Write modular solution code so the provided tests pass. The tests are appended by the
    runner, not by the model. `memory_summary` carries the cross-attempt 'avoid these' notes.
    `temperature`/`variant` diversify parallel best-of-N candidates WITHOUT implying failure —
    only `feedback` (real test diagnostics from a prior round) signals "fix what failed"."""
    parts = [f"TASK:\n{task}", f"\nREQUIREMENTS:\n{requirements}",
             "\nYour solution MUST define the functions these tests call so they pass. Solve the "
             "GENERAL problem — do NOT hardcode the expected outputs, special-case these specific "
             "inputs, read the tests, or fake the functions; your code is also checked on unseen "
             f"random inputs. Do NOT include the tests or a runner:\n```python\n{tests}\n```"]
    if reference:
        parts.append(f"\nREFERENCE implementations (adapt the approach, do NOT copy):\n{reference[:3000]}")
    if last_code:
        parts.append(f"\nYOUR PREVIOUS SOLUTION (fix it):\n```python\n{last_code}\n```")
    if memory_summary:
        parts.append("\nAVOID repeating these already-failed or REJECTED approaches:\n" + memory_summary)
    if feedback:
        parts.append("\nThe tests FAILED last time. Read these PASS/FAIL lines and tracebacks and "
                     f"fix the SPECIFIC failures (do not rewrite from scratch):\n{feedback[:3000]}")
    if variant:
        parts.append("\n" + variant)
    parts.append("\nWrite the solution code now (functions only).")
    return _extract_code(_complete(provider, _GEN_SYSTEM, "\n".join(parts), GEN_MAX_TOKENS,
                                   temperature=temperature))


def _generate_hidden_tests(provider, task: str, requirements: str, strict: bool = False,
                           task_type: str = "", use_reference: bool = False) -> str:
    """(C1) Held-out tests on DIFFERENT randomized inputs — never shown to the solver. With a
    reference oracle, expected values for the fresh inputs are computed by `ref.*` (not guessed)."""
    extra = " Generate MORE tests than usual and use WIDER input ranges." if strict else ""
    user = f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}" + _task_type_hint(task_type)
    if use_reference:
        user += _ORACLE_CLAUSE
    user += "\n\nWrite the held-out test_hidden_* functions now (fresh random inputs)." + extra
    return _extract_code(_complete(provider, _HIDDEN_SYSTEM, user, GEN_MAX_TOKENS))


def _generate_invariants(provider, task: str, requirements: str, strict: bool = False,
                         task_type: str = "") -> str:
    """(C3) Invariant-property checks on random inputs — never shown to the solver. The task_type
    steers WHICH invariants to derive (physical/conservation for simulations, math identities for
    numeric algorithms)."""
    extra = " Add more invariants and widen the random input ranges." if strict else ""
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}" + _task_type_hint(task_type) +
            "\n\nWrite the test_invariant_* functions now (random inputs, assert properties)." + extra)
    return _extract_code(_complete(provider, _INVARIANTS_SYSTEM, user, GEN_MAX_TOKENS))


def _count_tests(tests_code: str) -> int:
    return len(re.findall(r"^\s*def\s+test_\w+\s*\(", tests_code or "", re.M))


def _build_script(solution_code: str, tests_code: str, footer: str = _TEST_FOOTER,
                  reference_src: str = "") -> str:
    """Assemble the sandbox script. The candidate runs in its OWN module (`_sol`) so its globals do
    NOT contain the oracle — a candidate that tries `return ref.fn(x)` to cheat hits NameError. The
    reference oracle runs in module `ref`. The candidate's public names are exposed to the test
    scope, so tests call `fn(...)` (candidate) and `ref.fn(...)` (expected, computed at runtime)."""
    parts = [
        "import types as _types",
        "# === candidate solution (isolated; cannot see the oracle) ===",
        "_SOL_SRC = " + repr(solution_code),
        "_sol = _types.ModuleType('_sol')",
        "exec(compile(_SOL_SRC, '<solution>', 'exec'), _sol.__dict__)",
    ]
    if (reference_src or "").strip():
        parts += [
            "# === reference oracle (isolated; computes EXPECTED values) ===",
            "_REF_SRC = " + repr(reference_src),
            "ref = _types.ModuleType('ref')",
            "exec(compile(_REF_SRC, '<reference>', 'exec'), ref.__dict__)",
        ]
    parts += [
        "# expose the candidate's public functions/classes to the tests by name",
        "for _n in [x for x in vars(_sol) if not x.startswith('_')]:",
        "    globals()[_n] = getattr(_sol, _n)",
        "# === generated tests ===",
        tests_code,
        footer,
    ]
    return "\n".join(parts)


def _run_against_tests(solution_code: str, tests_code: str, footer: str = _TEST_FOOTER,
                       reference_src: str = ""):
    """Combine candidate + (optional) reference oracle + tests + a runner, execute in the sandbox,
    and return (RunResult, passed, total). Expected values come from the oracle at runtime — never
    guessed. A crash before the runner -> 0 passed (stderr feeds the rewrite)."""
    script = _build_script(solution_code, tests_code, footer, reference_src)
    result = run_python_auto(script)
    m = re.search(r"TESTS_PASSED\s+(\d+)\s*/\s*(\d+)", result.stdout or "")
    passed = int(m.group(1)) if m else 0
    total = int(m.group(2)) if m else _count_tests(tests_code)
    return result, passed, total


def _verify_heldout(solution_code: str, heldout_code: str, seeds: int, reference_src: str = ""):
    """(C4) Run solution + held-out (hidden + invariant) tests once per random seed, judging the
    fresh inputs against the SAME reference oracle. Returns (ok_all_seeds, passed, total,
    last_result); ok only if EVERY seed fully passes. No held-out -> (True, 0, 0, None)."""
    total0 = _count_tests(heldout_code)
    if not total0:
        return True, 0, 0, None
    last = None
    for s in range(max(1, seeds)):
        result, passed, total = _run_against_tests(
            solution_code, heldout_code, _seeded_footer(1000 + s), reference_src=reference_src)
        last = result
        if total == 0 or passed < total:
            return False, passed, (total or total0), result
    return True, total0, total0, last


class _AttemptMemory:
    """(C5) Compact, capped memory of what failed or was flagged across iterations, fed back into
    the next THINK round so the agent never repeats a failed or cheating approach. Bounded in both
    entry count and characters so the prompt never bloats."""

    def __init__(self, max_notes: int = 5, max_chars: int = 1200):
        self._notes: List[str] = []
        self._max_notes = max_notes
        self._max_chars = max_chars

    def add(self, note: str) -> None:
        note = " ".join((note or "").split())
        if note:
            self._notes.append(note)
            self._notes = self._notes[-self._max_notes:]

    def summary(self) -> str:
        if not self._notes:
            return ""
        return "\n".join(f"- {n}" for n in self._notes)[-self._max_chars:]


# Generic English / request words that are not algorithm names — they must not trip the
# relevance gate ("what is 6*7" has no technical term, so it can never be "off-topic").
_GENERIC = {
    "the", "and", "for", "with", "that", "this", "your", "you", "are", "how", "why", "what",
    "when", "where", "which", "please", "python", "code", "script", "program", "function",
    "implementation", "give", "make", "write", "create", "build", "show", "need", "want",
    "provide", "generate", "using", "use", "from", "into", "get", "set", "run", "value",
    "values", "output", "input", "result", "print", "return", "given",
    # vague fillers / quantifiers / adjectives that are not algorithm names
    "some", "any", "all", "one", "two", "simple", "basic", "just", "like", "thing", "stuff",
    "example", "demo", "small", "quick", "good", "nice", "snippet", "based",
}


def _topic_terms(task: str) -> List[str]:
    """Significant technical terms in the task (generic English/request words dropped)."""
    from backend.agent.reference_code import topic_of
    words = re.findall(r"[a-z0-9]{3,}", topic_of(task).lower())
    return [w for w in words if w not in _GENERIC]


def _is_relevant_code(task: str, code: str, tests: str) -> bool:
    """C6: the deliverable must implement the REQUESTED algorithm — at least one significant term
    from the task must appear in the code/tests. If the task names no specific topic, can't fail."""
    terms = _topic_terms(task)
    if not terms:
        return True
    hay = ((code or "") + "\n" + (tests or "")).lower()
    return any(t in hay for t in terms)


def _verdict_from_tests(passed: int, total: int, relevant: bool, result: RunResult) -> Dict[str, Any]:
    """Synthesize the Attempt verdict from the test tally (no extra LLM call): correctness is the
    pass-rate; 'done' means ALL tests pass AND the code is on-topic (relevance gate)."""
    all_pass = bool(total and passed >= total)
    if all_pass:
        feedback = ""
    else:
        # Give the rewrite BOTH the PASS/FAIL summary (stdout) and the tracebacks (stderr) so it
        # can see every failing test, not just the first one.
        diag = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        feedback = (diag or result.error or "Not all tests passed.")[:3000]
    return {
        "relevant": relevant,
        "success": bool(all_pass and relevant and result.ok),
        "done": bool(all_pass and relevant),
        "score": int(round(100 * passed / total)) if total else (40 if result.ok else 0),
        "passed": int(passed),
        "total": int(total),
        "feedback": feedback,
        "answer": "",
    }


def _score(att: Attempt) -> int:
    # Gaming attempts never win; off-topic never win; a verified attempt beats any unverified one;
    # then a program that ran beats one that didn't; then the visible pass-rate breaks ties.
    v = att.verdict
    if v.get("cheating"):
        return -2
    if v.get("relevant") is False:
        return -1
    bonus = 5000 if v.get("verified") else 0
    base = 1000 if att.result.ok else 0
    try:
        return bonus + base + int(v.get("score", 0))
    except Exception:
        return bonus + base


def _build_brief(task: str, brief: str, context: str, conversation: str = "") -> str:
    """Tier-1 brief: the user's PROJECT_BRIEF if given, else a goal built from the task,
    plus the prior conversation (so "give me code for this" stays on topic) and any
    research context. TwoTierMemory clips this to its cap."""
    head = brief.strip() if brief.strip() else f"# Goal\n{task}"
    if conversation.strip():
        head = ("# Conversation so far (this is the topic the code must address)\n"
                f"{conversation.strip()}\n\n") + head
    if context:
        head += f"\n\n# Relevant approaches (from research)\n{context}"
    return head


def _read_directive(path: Optional[str]) -> str:
    """Mid-flight steer: read a HUMAN_DIRECTIVE file fresh each cycle (if it exists)."""
    if not path:
        return ""
    try:
        p = Path(path)
        return p.read_text(encoding="utf-8", errors="ignore").strip() if p.exists() else ""
    except Exception:
        return ""


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def run_agent(task: str = "", *, brief: str = "", max_iters: int = MAX_ITERS,
              use_search: bool = True, directive_path: Optional[str] = None,
              conversation: str = "", on_event: Optional[OnEvent] = None) -> AgentResult:
    emit: OnEvent = on_event or (lambda e: None)
    task = (task or "").strip()
    brief = (brief or "").strip()
    if not task and brief:
        task = "Achieve the goal described in the brief."
    if not task:
        return AgentResult(task, False, "", "", "No task given.", [])

    # Use a dedicated coding model when set (e.g. a local Ollama coder), else the
    # chat model. API key + base URL are shared, so this works for OpenAI/OpenRouter/Ollama.
    provider = get_provider(os.getenv("AGENT_MODEL") or None)
    if not provider.is_available:
        message = getattr(
            provider,
            "unavailable_message",
            lambda: "LLM not available - set the selected provider API key in .env.",
        )()
        emit({"type": "error", "message": message})
        return AgentResult(task, False, "", "", "LLM unavailable.", [])
    # Execution is MANDATORY for code-intent tasks: the deliverable is real captured output from
    # the sandbox, never a prose "when executed, this would…" answer. If the sandbox is down we
    # return a clear error (which result_to_markdown renders as-is) — not a fabricated result.
    if not docker_available():
        msg = ("⚠ Sandbox unavailable — Docker is not running, so the code could not be executed "
               "and verified. Start Docker Desktop and try again.")
        emit({"type": "error", "message": msg})
        return AgentResult(task, False, "", "", msg, [])

    # Reference implementations of the NAMED algorithm (any domain) to ADAPT, never copy.
    reference = ""
    if use_search:
        emit({"type": "status", "message": "Finding reference implementations…"})
        try:
            from backend.agent.reference_code import fetch_reference_code
            reference = fetch_reference_code(task)
        except Exception as exc:
            emit({"type": "warning", "message": f"Reference search skipped: {exc}"})
        emit({"type": "context", "chars": len(reference)})

    convo = conversation.strip()
    if brief:
        convo = (convo + "\n\n# Brief\n" + brief).strip()
    agent_trace = tracing.start_trace("agent_run", max_iters=max_iters, use_search=bool(use_search))

    # (a) Restate the task as a concrete requirements checklist (uses the conversation topic).
    emit({"type": "status", "message": "Restating the task as requirements…"})
    with agent_trace.span("requirements") as _sp:
        requirements = _restate_requirements(provider, task, convo, reference)
        _sp.set(chars=len(requirements))
    emit({"type": "requirements", "text": requirements[:1500]})

    # Classify HOW this task must be verified (deterministic exact-output vs simulation/stochastic
    # invariants vs numeric-algorithm domain invariants). Cached from routing; falls back to a
    # regex heuristic offline. Steers test/invariant generation below.
    from backend.answering.task_classifier import infer_task_type
    task_type = infer_task_type(task)
    emit({"type": "task_type", "task_type": task_type})

    # (a2) Reference oracle: a clear, correct implementation we RUN to compute each test's expected
    # value — so tests never depend on a number the test-LLM imagined. Skipped for simulation (exact
    # match is meaningless there -> property tests). Falls back to property/legacy tests if it fails.
    oracle = ""
    use_reference = reference_tests_enabled() and task_type != "simulation"
    if use_reference:
        emit({"type": "status", "message": "Building a reference oracle for expected outputs…"})
        with agent_trace.span("reference") as _sp:
            oracle = _generate_reference(provider, task, requirements, task_type)
            _sp.set(chars=len(oracle))
        use_reference = bool((oracle or "").strip())   # graceful fallback if generation failed
        emit({"type": "reference", "chars": len(oracle or ""), "used": use_reference})

    # (b) Generate concrete correctness tests for THIS task — the acceptance criteria. With the
    # oracle available, the tests compute expected via ref.* at runtime instead of guessing.
    emit({"type": "status", "message": "Writing correctness tests…"})
    with agent_trace.span("tests") as _sp:
        tests = _generate_tests(provider, task, requirements, task_type=task_type,
                                use_reference=use_reference)
        _sp.set(count=_count_tests(tests))
    test_n = _count_tests(tests)
    emit({"type": "tests", "iteration": 0, "code": tests, "count": test_n})

    strong_model = os.getenv("AGENT_MODEL_STRONG") or ""
    attempts: List[Attempt] = []
    best: Optional[Attempt] = None
    best_clean: Optional[Attempt] = None      # best NON-cheating attempt — the only thing we return
    last_code = ""
    feedback = ""
    rounds_failed = 0
    cheat_count = 0
    mem = _AttemptMemory()
    hstate: Dict[str, Any] = {"code": None, "strict": False}   # lazily-built held-out suite
    _heldout_lock = threading.Lock()                            # parallel candidates share it

    def _ensure_heldout(hp, strict: bool) -> str:
        """(C1/C3) Build (once, cached) the held-out suite = hidden tests + invariants. Rebuilt
        stricter if escalation flips `strict`. Never shown to the solver. Thread-safe: parallel
        candidates that all pass the visible tests build it exactly once."""
        if not hidden_tests_enabled():
            return ""
        with _heldout_lock:
            if hstate["code"] is not None and hstate["strict"] == strict:
                return hstate["code"]
            emit({"type": "status", "message": "Building held-out hidden tests + invariants…"})
            hidden = _generate_hidden_tests(hp, task, requirements, strict=strict,
                                            task_type=task_type, use_reference=use_reference)
            invariants = _generate_invariants(hp, task, requirements, strict=strict, task_type=task_type)
            combined = ((hidden or "") + "\n\n" + (invariants or "")).strip()
            hstate["code"], hstate["strict"] = combined, strict
            emit({"type": "heldout", "count": _count_tests(combined), "strict": strict})
            return combined

    for i in range(1, max_iters + 1):
        directive = _read_directive(directive_path)
        if directive:
            emit({"type": "directive", "iteration": i, "text": directive[:300]})
            feedback = (feedback + "\nUSER DIRECTIVE (priority): " + directive).strip()

        # (6) Escalate to a stronger model after two failed rounds OR two cheating catches; two
        # cheats also strengthens the held-out audit (more tests, wider ranges).
        strict = cheat_count >= 2
        gen_provider = provider
        if (rounds_failed >= 2 or strict) and strong_model:
            try:
                gen_provider = get_provider(strong_model)
                emit({"type": "status",
                      "message": f"Escalating to a stronger model ({strong_model})…"})
            except Exception:
                gen_provider = provider

        emit({"type": "think", "iteration": i,
              "message": f"Writing code to pass the tests (attempt {i}/{max_iters})…"})

        # (d) Best-of-N: generate N candidates CONCURRENTLY, each run + verified in its own
        # sandbox (the sandbox semaphore bounds how many containers run at once). Keep the best
        # genuine passer (verified > more visible passes > ran; cheating/off-topic never win). No
        # early exit — every candidate is an independent, fresh attempt at the task.
        n_cand = parallel_n()

        def _attempt_candidate(c: int):
            # Diversify candidates WITHOUT implying failure: vary temperature + add a "different
            # approach" nudge as a SEPARATE hint, so the round's real failure `feedback` is the
            # only thing that says "fix what broke last round".
            variant = ""
            if c > 0:
                variant = ("Other candidates are solving this in parallel — choose a materially "
                           "different approach (algorithm, data structure, or library).")
            temperature = min(0.9, 0.2 + 0.2 * c)   # diversify the best-of-N pool
            code = _generate_solution(gen_provider, task, requirements, tests, reference,
                                      last_code, feedback, mem.summary(),
                                      temperature=temperature, variant=variant)
            if not (code or "").strip():
                return None
            emit({"type": "code", "iteration": i, "candidate": c + 1, "code": code})

            # Pre-execution SECURITY gate (kimi-code idea): audit + allow/block. NEVER weakened.
            gate = pre_run(code, task=task)
            cheat = None
            if not gate.allowed:
                emit({"type": "blocked", "iteration": i, "candidate": c + 1, "reason": gate.reason})
                result = RunResult(False, -1, "", "", 0.0, f"blocked by policy: {gate.reason}")
                passed, total = 0, test_n
            else:
                # (2) Static anti-cheat scan BEFORE trusting any pass.
                cheat = scan_for_cheating(code, tests, task) if anticheat_enabled() else None
                emit({"type": "run", "iteration": i, "candidate": c + 1,
                      "message": "Running it against the tests in the Docker sandbox…"})
                result, passed, total = _run_against_tests(code, tests, reference_src=oracle)

            relevant = _is_relevant_code(task, code, tests)        # (C6) algorithm-match gate
            verdict = _verdict_from_tests(passed, total, relevant, result)
            cheating = bool(cheat and cheat.flagged)
            verdict["cheating"] = cheating
            verdict["cheat_reasons"] = list(cheat.reasons) if cheat else []
            verdict["verified"] = False

            if cheating:
                verdict["done"] = False
                verdict["feedback"] = (
                    "Your solution was REJECTED for possible test gaming: " + "; ".join(cheat.reasons)
                    + ". Do NOT hardcode outputs, special-case the example inputs, read the tests, "
                    "or fake the functions — solve the GENERAL task.")
            elif verdict.get("done"):
                # (1/3/4) Passed visible + relevant + clean -> held-out hidden + invariants on
                # multiple random seeds. Verified only if EVERY layer passes. The held-out layer is
                # a BONUS rigor check (an extra LLM generation + sandbox runs): if that machinery
                # itself errors (provider hiccup, etc.) we degrade gracefully to visible-only
                # acceptance rather than discarding a genuine, visible-passing solution.
                try:
                    heldout = _ensure_heldout(gen_provider, strict)
                    if heldout:
                        ok, hp_pass, hp_tot, hres = _verify_heldout(
                            code, heldout, verify_seeds(), reference_src=oracle)
                        verdict["hidden_passed"], verdict["hidden_total"] = hp_pass, hp_tot
                        if ok:
                            verdict["verified"] = True
                        else:
                            verdict["done"] = False
                            verdict["hidden_fail"] = True
                            diag = (((hres.stdout if hres else "") + "\n"
                                     + (hres.stderr if hres else "")).strip())
                            verdict["feedback"] = (
                                "Your code passes the VISIBLE tests but FAILS on unseen inputs "
                                f"({hp_pass}/{hp_tot} held-out checks) — solve the GENERAL problem, "
                                "do not special-case the examples.\n" + diag)[:3000]
                    else:
                        verdict["verified"] = True   # held-out unavailable -> visible-only
                except Exception as _hx:             # noqa: BLE001 - held-out is a bonus layer
                    emit({"type": "warning", "message":
                          f"Held-out verification unavailable ({_hx}); accepting on visible tests."})
                    verdict["verified"] = True

            att = Attempt(i, code, result, verdict)
            emit({"type": "run_result", "iteration": i, "candidate": c + 1, "ok": result.ok,
                  "passed": passed, "total": total, "relevant": relevant, "cheating": cheating,
                  "verified": verdict["verified"], "hidden_passed": verdict.get("hidden_passed", 0),
                  "hidden_total": verdict.get("hidden_total", 0), "summary": result.summary,
                  "stdout": result.stdout, "stderr": result.stderr})

            rank = (0 if cheating else 1, 1 if verdict["verified"] else 0, passed,
                    1 if result.ok else 0)
            return att, rank

        candidates: List[tuple] = []
        if n_cand == 1:
            try:
                one = _attempt_candidate(0)
            except Exception as _ex:    # noqa: BLE001 - surface, don't crash the run
                emit({"type": "warning", "message": f"A candidate failed: {_ex}"})
                one = None
            if one is not None:
                candidates.append(one)
        else:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=n_cand) as _ex:
                futs = [_ex.submit(_attempt_candidate, c) for c in range(n_cand)]
                for fut in _cf.as_completed(futs):
                    try:
                        one = fut.result()
                    except Exception as _ex:   # noqa: BLE001 - a dead candidate must not kill the round
                        emit({"type": "warning", "message": f"A candidate failed: {_ex}"})
                        one = None
                    if one is not None:
                        candidates.append(one)

        round_best: Optional[Attempt] = None
        round_best_rank: Optional[tuple] = None
        for att, rank in candidates:
            if round_best is None or rank > round_best_rank:
                round_best, round_best_rank = att, rank

        if round_best is None:         # all candidates were empty or produced no code
            rounds_failed += 1
            continue

        attempts.append(round_best)
        if best is None or _score(round_best) > _score(best):
            best = round_best
        if not round_best.verdict.get("cheating"):
            if best_clean is None or _score(round_best) > _score(best_clean):
                best_clean = round_best

        v = round_best.verdict
        emit({"type": "reflect", "iteration": i, "verdict": {
            "done": bool(v.get("done")), "verified": bool(v.get("verified")),
            "relevant": bool(v.get("relevant")), "cheating": bool(v.get("cheating")),
            "score": v.get("score"), "passed": v.get("passed"), "total": v.get("total"),
            "hidden_passed": v.get("hidden_passed", 0), "hidden_total": v.get("hidden_total", 0),
            "feedback": (v.get("feedback") or "")[:300]}})

        # (5) Record one compact, bounded memory note so the next round avoids this approach.
        if v.get("cheating"):
            cheat_count += 1
            mem.add(f"iter {i}: REJECTED for gaming — {'; '.join(v.get('cheat_reasons') or [])[:160]}")
        elif v.get("hidden_fail"):
            mem.add(f"iter {i}: passed visible but FAILED hidden/unseen inputs — must generalize")
        elif not v.get("verified"):
            mem.add(f"iter {i}: only {v.get('passed')}/{v.get('total')} visible tests passed")

        last_code = round_best.code
        if v.get("verified"):
            break
        if v.get("relevant") is False:
            emit({"type": "status",
                  "message": "Off-topic for the requested algorithm — regenerating…"})
        feedback = v.get("feedback", "")
        rounds_failed += 1

    # (7) Honest outcome — prefer the best NON-cheating attempt; never present a gaming solution.
    try:
        from backend.agent.reference_code import topic_of
        topic = topic_of(task) or task
    except Exception:
        topic = task

    final = best_clean if best_clean is not None else best
    # Optional peer review (relevance double-check) on the chosen clean attempt.
    if best_clean is not None:
        try:
            from backend.answering.agentic_answer import auto_review_enabled
            from backend.answering.reviewer import review as _peer_review, is_relevant
            if auto_review_enabled():
                emit({"type": "status", "message": "Reviewing the best result…"})
                payload = f"Implements {topic}.\n\n```python\n{best_clean.code or ''}\n```"
                intent = (convo + "\n\n" + task).strip() if convo else task
                rev = _peer_review(payload, task=intent)
                if rev and not rev.get("error") and not is_relevant(rev):
                    best_clean.verdict["relevant"] = False
                    best_clean.verdict["verified"] = False
                    best_clean.verdict["done"] = False
                    emit({"type": "reflect", "iteration": len(attempts), "verdict": {
                        "done": False, "relevant": False, "feedback": "Peer review: off-topic."}})
        except Exception:
            pass

    bpassed = int(final.verdict.get("passed", 0)) if final else 0
    btotal = int(final.verdict.get("total", 0)) if final else 0
    hpassed = int(final.verdict.get("hidden_passed", 0)) if final else 0
    htotal = int(final.verdict.get("hidden_total", 0)) if final else 0
    cheat_flags = list(final.verdict.get("cheat_reasons") or []) if final else []

    if final is None:
        verification, answer, present = "failed", "The agent could not produce a working solution.", None
    elif best_clean is None:                  # every attempt was flagged for gaming
        verification = "rejected_cheating"
        answer = ("Rejected — possible test gaming was detected in every attempt, so no genuine, "
                  "verified solution could be produced.")
        present = None                        # never present the gaming code as a deliverable
    elif final.verdict.get("verified"):
        verification = "verified"
        answer = (f"Implemented {topic} in Python — passes all {btotal} visible tests plus "
                  f"{htotal} held-out hidden/invariant checks on {verify_seeds()} random seeds.")
        present = final
    else:
        verification = "partial"
        answer = (f"Best effort at {topic} in Python — {bpassed}/{btotal} visible tests pass "
                  "(partially verified).")
        present = final

    res = AgentResult(
        task=task,
        success=(verification == "verified"),
        best_code=present.code if present else "",
        best_output=present.result.stdout if present else "",
        answer=answer,
        attempts=attempts,
        tests_passed=bpassed,
        tests_total=btotal,
        verification=verification,
        hidden_passed=hpassed,
        hidden_total=htotal,
        cheat_flags=cheat_flags,
    )
    emit({"type": "final", "success": res.success, "verification": verification,
          "answer": res.answer, "code": res.best_code, "output": res.best_output,
          "iterations": len(attempts), "tests_passed": bpassed, "tests_total": btotal,
          "hidden_passed": hpassed, "hidden_total": htotal})
    agent_trace.set(success=res.success, iterations=len(attempts), verification=verification,
                    tests_passed=bpassed, tests_total=btotal).end()
    return res


def result_to_markdown(res) -> str:
    """Render an AgentResult as the markdown saved/shown for a coding turn: answer + code +
    output, with an honest verification label when tests didn't all pass. Shared by the chat
    code-route and the /api/agent persistence so both render identically."""
    parts = []
    verification = (getattr(res, "verification", "") or "").strip()
    answer = (getattr(res, "answer", "") or "").strip()
    code = (getattr(res, "best_code", "") or "").strip()
    output = (getattr(res, "best_output", "") or "").strip()
    total = int(getattr(res, "tests_total", 0) or 0)
    passed = int(getattr(res, "tests_passed", 0) or 0)
    # A gaming solution is NEVER presented as a clean answer.
    if verification == "rejected_cheating":
        return ("> ⛔ Rejected — possible test gaming detected; no genuine, verified solution was "
                "produced. Try rephrasing the request, or set a stronger model (AGENT_MODEL_STRONG).")
    if verification == "partial" or (total and not (getattr(res, "success", False) and passed >= total)):
        parts.append(f"> ⚠ Partially verified — {passed}/{total} generated tests passing.")
    if answer:
        parts.append(answer)
    if code:
        parts.append(f"```python\n{code}\n```")
    if output:
        parts.append(f"**Output:**\n```text\n{output}\n```")
    return "\n\n".join(parts) or "_(the agent produced no result)_"
