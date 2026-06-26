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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.agent.anticheat import anticheat_enabled, scan_for_cheating
from backend.agent.code_runner import RunResult, clip_keep_ends, docker_available, run_python_auto
from backend.common import request_context as _rc

# Demo stdout kept for the completeness gate + the user-facing Output block. Head+tail (not
# head-only) so a requested value printed AFTER a large intermediate dump still survives.
DEMO_OUTPUT_CAP = 12_000
from backend.agent.hooks import pre_run
from backend.llm.streaming_provider import CATALOG, DEFAULT_OPENAI_MODEL, get_provider
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


def max_attempts() -> int:
    """Live read (AGENT_MAX_ATTEMPTS, default 10, clamped >=1): the upper bound on generate->verify
    rounds. The loop keeps iterating — feeding each round's GENUINE failing checks back — until the
    solution is fully verified, this bound is hit, or it STALLS (no progress). Falls back to the
    legacy AGENT_MAX_ITERS if AGENT_MAX_ATTEMPTS is unset, so existing configs keep working."""
    raw = os.getenv("AGENT_MAX_ATTEMPTS") or os.getenv("AGENT_MAX_ITERS") or "10"
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 10


def stall_limit() -> int:
    """Live read (AGENT_STALL_LIMIT, default 3, clamped >=1): stop early after this many CONSECUTIVE
    rounds that fail to reduce the number of genuine failing checks. Prevents looping to the attempt
    cap when more attempts clearly are not helping — return the best effort, labelled honestly."""
    try:
        return max(1, int(os.getenv("AGENT_STALL_LIMIT", "3")))
    except (TypeError, ValueError):
        return 3


def reference_tests_enabled() -> bool:
    """Live read (AGENT_REFERENCE_TESTS, default on): derive each test's EXPECTED value by RUNNING
    a reference oracle in the sandbox, instead of letting the test-LLM guess literal outputs. Off
    falls back to property/legacy tests."""
    return os.getenv("AGENT_REFERENCE_TESTS", "true").strip().lower() not in ("0", "false", "no", "off")


def delivery_gates_enabled() -> bool:
    """Live read (AGENT_DELIVERY_GATES, default on): for a task that asks to print/show/return a
    result, enforce the EXECUTION gate (the solution must produce real stdout) and the COMPLETENESS
    gate (every requested deliverable must appear in that stdout) before labelling a solution
    'verified'. Off skips these two gates (visible + held-out only)."""
    return os.getenv("AGENT_DELIVERY_GATES", "true").strip().lower() not in ("0", "false", "no", "off")


def definition_gate_enabled() -> bool:
    """Live read (AGENT_DEFINITION_GATE, default on): held out alongside the hidden tests, one check
    PER requested output asserts the REPORTED value is the EXACT quantity the user asked for — right
    quantity, point/time, aggregation, units — computed independently of the candidate AND the
    reference oracle (which could share a wrong definition). Catches 'right logic, wrong answer'."""
    return os.getenv("AGENT_DEFINITION_GATE", "true").strip().lower() not in ("0", "false", "no", "off")


def test_validation_enabled() -> bool:
    """Live read (AGENT_TEST_VALIDATION, default on): before a generated test is allowed to FAIL a
    candidate, run it against the known-correct reference ORACLE. Any test the oracle itself fails is
    INVALID — its expected value was guessed, its tolerance is too tight for the method, or it checks
    the wrong quantity — and is QUARANTINED: excluded from the candidate's pass/total so a flawed test
    can never falsely fail correct code, while every test the oracle PASSES still gates genuinely wrong
    code. Needs a reference oracle (AGENT_REFERENCE_TESTS); off, every generated test is trusted."""
    return os.getenv("AGENT_TEST_VALIDATION", "true").strip().lower() not in ("0", "false", "no", "off")


def test_critic_enabled() -> bool:
    """Live read (AGENT_TEST_CRITIC, default on): before any generated test is allowed to judge a
    candidate, a SEPARATE TEST-CRITIC role audits the suite and REWRITES every invalid test (guessed
    value / hardcoded tolerance / exact-match on a non-unique quantity / wrong operational definition /
    requirement-not-implied-by-task / wrong entity) into a check of the TRUE requirement, keeping valid
    tests verbatim. Complements the execution-based quarantine (which only DROPS oracle-failing tests):
    the critic repairs the gap a drop leaves and catches invalid checks an oracle itself shares. Off ->
    only the inline execution quarantine runs."""
    return os.getenv("AGENT_TEST_CRITIC", "true").strip().lower() not in ("0", "false", "no", "off")


def nonunique_validation_enabled() -> bool:
    """Live read (AGENT_NONUNIQUE_VALIDATION, default on): extend test-validation to catch tests that
    assert EXACT equality on a NON-UNIQUE quantity — one defined only up to scaling / sign / ordering /
    phase / basis / representation, or produced by an underdetermined procedure. The single oracle
    trivially agrees with itself, so such a test slips past the normal quarantine yet fails a different
    VALID solution. We detect it by EXECUTION: an independent cross-reference that returns a different
    valid representation fails exactly those tests while still satisfying every property check, so they
    are quarantined. Needs the reference oracle + test-validation; fail-open."""
    return os.getenv("AGENT_NONUNIQUE_VALIDATION", "true").strip().lower() not in ("0", "false", "no", "off")


def root_cause_enabled() -> bool:
    """Live read (AGENT_ROOT_CAUSE_DIAGNOSIS, default on): before each rewrite, DIAGNOSE why a check
    failed — map the SYMPTOM (a drifting invariant, a wrong sign, a blow-up) to the likely MECHANISM
    and code location — and feed that targeted diagnosis to the next attempt, instead of mutating the
    code blindly. Off falls back to the raw PASS/FAIL + traceback feedback only."""
    return os.getenv("AGENT_ROOT_CAUSE_DIAGNOSIS", "true").strip().lower() not in ("0", "false", "no", "off")


def result_memory_enabled() -> bool:
    """Live read (AGENT_RESULT_MEMORY, default on): record every code-agent run's outcome to the
    SQLite store (for the developer failure-pattern report) and, on a new task, SEED the first attempt
    with a near-identical VERIFIED prior solution. Reuse never bypasses a gate; only verified runs are
    reused; the agent never edits its own source. Off disables both recording and reuse."""
    return os.getenv("AGENT_RESULT_MEMORY", "true").strip().lower() not in ("0", "false", "no", "off")


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
    attempts_taken: int = 0           # generate->verify rounds actually run
    stop_reason: str = ""             # verified | stall | max_attempts


# ----------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------
def _complete(provider, system: str, user: str, max_tokens: int, temperature: float = 0.2) -> str:
    return "".join(provider.stream_chat(
        [{"role": "user", "content": user}], system=system,
        max_tokens=max_tokens, temperature=temperature,
    )).strip()


# ----------------------------------------------------------------------
# Resilient model selection: respect the user's choice, retry transient provider errors, and fall
# back to another AVAILABLE model instead of failing the whole request when one is rate-limited.
# ----------------------------------------------------------------------
_TRANSIENT_MARKERS = ("rate", "quota", "429", "resource_exhausted", "timeout", "timed out",
                      "deadline", "unavailable", "503", "502", "500", "overloaded", "connection",
                      "temporarily", "try again", "apiconnection", "apitimeout")


def _is_transient_err(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in _TRANSIENT_MARKERS)


def _user_selected_model() -> bool:
    """True when the user explicitly chose a model — the agent must NOT override it with a different
    provider on escalation. Set AGENT_MODEL, or a chat model different from the built-in default."""
    if (os.getenv("AGENT_MODEL") or "").strip():
        return True
    return (_rc.request_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) or "").strip() != DEFAULT_OPENAI_MODEL


def _model_available(model: str) -> bool:
    try:
        return bool(get_provider(model).is_available)
    except Exception:
        return False


def _agent_model_chain() -> List[str]:
    """Ordered, deduped list of AVAILABLE model ids, best-first: the user's selection first
    (AGENT_MODEL or the chat's OPENAI_MODEL), then the configured stronger model, then any other
    configured catalog model — a resilient fallback chain so a 429/timeout on one model switches to
    another instead of failing the request."""
    primary = (os.getenv("AGENT_MODEL") or _rc.request_str("OPENAI_MODEL", "") or DEFAULT_OPENAI_MODEL).strip()
    chain: List[str] = [primary] if primary else []      # the user's choice ALWAYS comes first
    strong = (os.getenv("AGENT_MODEL_STRONG") or "").strip()
    for m in ([strong] if strong else []) + [mid for mid, *_ in CATALOG]:
        m = (m or "").strip()
        if m and m not in chain and _model_available(m):
            chain.append(m)
    return chain or [primary or DEFAULT_OPENAI_MODEL]


def _escalated_chain(primary_chain: List[str]) -> List[str]:
    """Chain when escalating after failed rounds. RESPECTS the user's choice: if a model was
    explicitly selected we keep the user's chain (never switch providers). Only with NO selection do
    we let AGENT_MODEL_STRONG lead, keeping the rest as fallback."""
    strong = (os.getenv("AGENT_MODEL_STRONG") or "").strip()
    if not strong or _user_selected_model() or not _model_available(strong):
        return list(primary_chain)
    return [strong] + [m for m in primary_chain if m != strong]


class ResilientProvider:
    """Wraps an ordered list of model ids and presents the LLMProvider interface. On any provider
    error (rate limit, timeout, 5xx, auth, connection) it retries with backoff, then FALLS BACK to
    the next available model — emitting a clear note — so one rate-limited provider never fails a
    request when another configured model works. Each model's full response is buffered before
    yielding, so a fallback never produces partial/duplicated output. (Agent use only — the live
    chat path streams directly.)"""

    name = "resilient"

    def __init__(self, models: List[str], emit: Optional[OnEvent] = None, max_retries: int = 3):
        self._models = list(dict.fromkeys(m for m in (models or []) if m)) or [DEFAULT_OPENAI_MODEL]
        self._emit = emit or (lambda e: None)
        self._max_retries = max(1, max_retries)
        self._providers: Dict[str, Any] = {}
        self._active = 0                                 # index of the last model that worked

    def _provider(self, model: str):
        if model not in self._providers:
            self._providers[model] = get_provider(model)
        return self._providers[model]

    @property
    def model(self) -> str:
        return self._models[self._active] if self._models else ""

    @property
    def is_available(self) -> bool:
        return any(self._provider(m).is_available for m in self._models)

    def unavailable_message(self) -> str:
        return ("No configured LLM is available — set a provider API key in .env "
                "(GEMINI_API_KEY, MISTRAL_API_KEY, or OPENAI_CLOUD_KEY).")

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3,
                    yield_reasoning=False):
        # Available models in preference order, starting from the last one that worked.
        order = list(range(self._active, len(self._models))) + list(range(0, self._active))
        avail = [self._models[i] for i in order if self._provider(self._models[i]).is_available]
        if not avail:
            raise RuntimeError(self.unavailable_message())
        last_err: Optional[Exception] = None
        for pos, model in enumerate(avail):
            prov = self._provider(model)
            delay = 1.0
            for attempt in range(self._max_retries):
                try:
                    chunks = list(prov.stream_chat(           # buffer fully before yielding
                        messages, system=system, max_tokens=max_tokens,
                        temperature=temperature, yield_reasoning=yield_reasoning))
                    self._active = self._models.index(model)  # prefer this model next time
                    for c in chunks:
                        yield c
                    return
                except Exception as e:                        # noqa: BLE001 - classify by message
                    last_err = e
                    if _is_transient_err(str(e)) and attempt < self._max_retries - 1:
                        time.sleep(min(delay, 8.0))
                        delay *= 2
                        continue
                    break                                     # non-transient / exhausted -> next model
            if pos + 1 < len(avail):
                self._emit({"type": "warning", "message":
                            f"Model {model} unavailable ({type(last_err).__name__}); "
                            f"switching to {avail[pos + 1]}…"})
        if last_err is not None:
            raise last_err
        return


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
    "each with a clear signature. A separate harness IMPORTS your functions and exercises them with "
    "unit tests, so do NOT put bare top-level prints or a test runner at module scope (they would run "
    "on import). WHEN THE TASK ASKS TO COMPUTE AND PRINT / REPORT / SHOW RESULTS, your file must ALSO "
    "be a RUNNABLE PROGRAM: add an `if __name__ == \"__main__\":` block that CALLS your functions on "
    "the task's inputs and PRINTS every requested value with a clear label — so running the file "
    "actually produces those outputs, while importing it for the tests does not run that block. "
    "Defining functions with NO call site is an INCOMPLETE program for an output task. "
    "Write code CORRECT BY DESIGN — right for ANY valid input, not just the value it will be demoed "
    "on. Concretely: (1) make every input assumption EXPLICIT and enforce it — units (e.g. radians "
    "vs degrees before a trig call), valid ranges, types, array shapes, and indexing convention: "
    "CONVERT to the form you need, or `raise` a clear ValueError on a violation — NEVER silently "
    "assume the caller matched your convention; (2) handle the edge/boundary cases the task implies "
    "and that are valid for it (empty, zero, negative, single element, min/max); (3) use NO magic "
    "constants that only work for the example — derive everything from the inputs. A function that "
    "returns the right number for the demo value but breaks on another valid input is WRONG. "
    "CONSUME RETURN VALUES BY THEIR TRUE CONTRACT: when you call any function — your own or a "
    "library's — use what it returns by its ACTUAL structure (type, array shape and axis meaning, "
    "tuple/dict arity and field order, length, units). NEVER index, slice, unpack, or iterate a "
    "returned value as if it had a different structure than the function returns — e.g. do not slice "
    "a fixed-size summary/stats tuple as if it were the full data array, treat a scalar as a "
    "sequence, read the wrong axis, or unpack the wrong number of values. At every boundary where a "
    "produced value feeds the next step, make a structural mismatch fail LOUDLY — assert the expected "
    "type/shape/length/arity (or convert) — instead of silently computing on the wrong data. "
    "EXPOSE EVERY REPORTED QUANTITY: if the task asks to report an INTERMEDIATE or COMPONENT value "
    "(an intermediate signal or its envelope, a sub-result, a per-stage metric), the function that "
    "computes it must RETURN it (or otherwise make it directly available) — do NOT compute a reported "
    "intermediate inside a function and discard it, leaving only the final result. When you print or "
    "report a value, take it FROM the real return of the function that produced it; NEVER re-derive a "
    "substitute, approximate it a second time, or report a different-but-related quantity (e.g. the "
    "final signal in place of the intermediate envelope) in its place — a fabricated reported value is "
    "WRONG even when the core computation is right. "
    "FOLLOW THE REQUEST'S EXPLICIT INSTRUCTIONS LITERALLY — do NOT substitute your own plausible "
    "choices. (1) PRINT EXACTLY THE NAMED OUTPUTS: compute and print the precise quantities the request "
    "names, by their stated meaning — if it asks for 'the sum of X' print the sum of X (NOT the peak of "
    "X, NOT a per-component breakdown); print every named output, the right quantity correctly "
    "aggregated, and do not silently report a different, related set in their place. (2) USE THE "
    "SPECIFIED METHOD: if the request names HOW — a method, operation, algorithm, or approach (e.g. "
    "'apply the filter using convolution', 'integrate with RK4', 'via the FFT') — implement it with "
    "THAT method; do NOT swap in a different technique because it seems equivalent (a causal/recursive "
    "filter and a direct convolution differ in phase/delay, so they are NOT interchangeable). (3) "
    "IMPLEMENT THE REQUEST'S EXACT FORMULA/DEFINITION when it gives one, not a textbook variant. A "
    "result that reports DIFFERENT quantities or uses a DIFFERENT method than the request specified is "
    "WRONG even if its own internal logic is self-consistent. "
    "Deliver the COMPLETE task: implement EVERY function and compute EVERY result the request asks "
    "for — never a subset. "
    "At RUNTIME the sandbox has no network, no file access, and no input() — do not use "
    "them (third-party imports are fine). The code must run to completion in a few seconds. "
    "Output ONLY the Python code — no explanation, no markdown."
)

_REQ_SYSTEM = (
    "You are a senior engineer. Restate the user's coding task as a short, concrete checklist of "
    "requirements (3-7 bullets): the function(s) to implement WITH their signatures, the inputs "
    "and outputs, and the key correctness properties to satisfy. List EVERY explicit DELIVERABLE the "
    "request asks for — each value to print/return, each comparison, each property to verify, "
    "INCLUDING any INTERMEDIATE or COMPONENT quantity it asks to report (an intermediate signal or "
    "envelope, a sub-result, a per-stage metric) — so nothing requested is dropped. State which "
    "function must EXPOSE (return) each reported quantity, so an intermediate needed for the report is "
    "a RETURN, not a value computed then discarded. For each deliverable, pin its EXACT DEFINITION: "
    "the precise "
    "quantity (e.g. median NOT mean, diameter NOT radius), how it is AGGREGATED (sum / mean / max / "
    "last), the POINT/INDEX it is taken at (e.g. initial = at the start / t=0 / step 0 / index 0; "
    "final = at the end), and the units/convention — so a related-but-different quantity, a wrong "
    "aggregation, or a value at the wrong point is caught as WRONG. Include the INPUT CONTRACT "
    "explicitly — the units, valid ranges, types/shapes, and indexing convention each argument must "
    "satisfy — so the implementation and the tests can ENFORCE it rather than guess. State each "
    "function's RETURN CONTRACT just as precisely — the return type, and for an array its shape/"
    "dimensions and what each axis means, for a tuple/dict its arity and the name and ORDER of every "
    "field, plus units — so a consumer cannot misread the structure of what it returns. "
    "If the request SPECIFIES HOW to do something — a NAMED METHOD, operation, algorithm, or approach "
    "(e.g. 'apply the filter using convolution', 'integrate with RK4', 'compute via the FFT', 'sort "
    "with mergesort') — list that as a MUST-USE requirement: the solution must use THAT exact method, "
    "not a different-but-equivalent-looking one (a different technique can give different results — a "
    "causal/recursive filter vs a direct convolution differ in phase/delay). If the request GIVES a "
    "formula or DEFINES a term, restate that EXACT formula/definition as the one to implement, not a "
    "textbook variant. Use the "
    "conversation context if given. Output ONLY the bullet list."
)

_REFERENCE_SYSTEM = (
    "You write a CLEAR, CORRECT REFERENCE implementation that serves as the trusted ORACLE for "
    "testing: its EXECUTED outputs are the expected values a separate candidate solution is "
    "compared against. Correctness matters far more than speed or elegance — use the simplest "
    "approach you are SURE is right. Model the SPECIFIC system the task describes, INCLUDING every term "
    "or condition that breaks an idealised behaviour (a dissipative / damping term, a driving / forcing "
    "/ source term, an open boundary, an injection / removal process, an explicit asymmetry, randomness "
    "/ noise) — do NOT silently 'simplify' to a textbook ideal (an UNDAMPED oscillator for a damped "
    "one, a CLOSED system for an open one); a reference that drops such a term is WRONG and rejects "
    "correct code. If the request SPECIFIES the method / operation / algorithm to use (e.g. 'apply the "
    "filter using convolution', 'integrate with RK4', 'via the FFT'), your reference MUST use THAT "
    "EXACT method — the spec is authoritative, so the oracle encodes the required approach and a "
    "candidate that substitutes a different method is caught by the value mismatch. ONLY when the "
    "request leaves the method OPEN, prefer a DIFFERENT, INDEPENDENT method from what an optimized "
    "candidate would write (e.g. a closed-form formula vs a numerical loop, a brute-force definition "
    "vs an optimized algorithm) so that if the candidate makes a hidden wrong assumption your oracle "
    "does NOT share it and the mismatch is caught instead of agreed on. "
    "Implement the SAME function name(s) and signature(s) the task "
    "/ requirements specify, so the candidate can be called identically. You MAY use numpy / scipy "
    "/ stdlib. Define ONLY the functions — no __main__, no prints, no tests. At runtime there is no "
    "network, no file access, no input(). Output ONLY the Python code — no explanation, no markdown."
)

_DRIVER_SYSTEM = (
    "You write a SHORT driver snippet that DEMONSTRATES a finished solution: it calls the solution's "
    "already-defined functions on representative inputs taken from the task and PRINTS the results "
    "with clear labels (e.g. print('period (s):', period)). Print EVERY value the request asks for — "
    "each requested deliverable on its own line with a clear text label — so every one is visible in "
    "the output. Obtain each reported value from the solution's REAL function return that produced it "
    "(call the function that exposes it); NEVER re-derive, approximate, or substitute a requested "
    "value — especially an intermediate (an envelope, a sub-result): if it is not available from the "
    "functions, that is the solution's bug to fix, not something to fabricate here.\n"
    "DO NOT FLOOD STDOUT: never print a whole large array, matrix, tensor, DataFrame, or dataset in "
    "full. For any large or collection value, print only a COMPACT SUMMARY — its shape/length plus a "
    "few elements, or a statistic (min/max/mean) — NEVER thousands of values. Print ONLY the "
    "requested deliverables and such summaries; do NOT dump unrelated intermediate or debug data.\n"
    "PRINT THE REQUESTED RESULTS LAST: emit the specific value(s) the user asked for in a clear FINAL "
    "block at the very END, one labelled value per line, so they are ALWAYS visible even if other "
    "prints appear earlier and even if earlier output is long.\n"
    "The solution is ALREADY DEFINED above your snippet — do NOT redefine it, do NOT write tests, add "
    "an import only if truly needed. Keep it under ~20 lines and a couple of seconds to run. Output "
    "ONLY the snippet code."
)

_TESTS_SYSTEM = (
    "You write rigorous but ROBUST unit tests as plain Python (no pytest, no unittest). Given a "
    "task and its requirements, write 5-7 focused test functions named test_<name>() that:\n"
    "- call the SOLUTION's functions directly (they are defined in the test scope);\n"
    "- use SELF-CONSISTENT inputs: the SAME calling convention and array shapes in EVERY test, "
    "matching ONE function signature — do NOT require the function to accept several input shapes;\n"
    "- compare floats with tolerances (math.isclose, or numpy.allclose with explicit rtol/atol), "
    "NEVER exact ==; build small CONCRETE inputs (use numpy if it helps);\n"
    "- DERIVE every tolerance FROM THE PROBLEM — never paste an arbitrary fixed threshold. For a "
    "STOCHASTIC / Monte-Carlo result, a correct estimate is EXPECTED to differ from the true value "
    "by about one STANDARD ERROR, so compare within a few SE (estimate SE from the sample variance "
    "and N, e.g. atol ~= k * stdev / sqrt(N) with k=3-5) — a tight atol on a noisy mean is wrong and "
    "WILL falsely fail correct code. For an ITERATIVE / NUMERICAL method, size the tolerance to the "
    "method's own error (step size h, discretization, or the convergence tol). Only a DETERMINISTIC "
    "closed-form result earns a tight tolerance (~1e-9). A tolerance too tight for the method is the "
    "#2 cause of false failures;\n"
    "- obtain every EXPECTED value by COMPUTING it — from the reference oracle when one is provided "
    "(see below), otherwise as a PROPERTY that holds for any correct implementation. NEVER "
    "hand-write a literal expected number/output you imagined; a guessed expected value is the #1 "
    "cause of false failures;\n"
    "- test for the EXACT quantities the request NAMES (not a related substitute), and if the request "
    "SPECIFIES a METHOD / operation / approach, derive the expected value USING that method so code "
    "that used a DIFFERENT technique is caught;\n"
    "- CONSUME RETURN VALUES BY THEIR TRUE CONTRACT: whatever you read from the solution or from "
    "`ref.*`, use it by its ACTUAL structure (type, shape, arity, field order) — never slice, unpack, "
    "or iterate it as a different shape — and assert that structure before comparing values, so your "
    "check operates on the RIGHT object, not a wrong-shaped stand-in;\n"
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
    "data, and do NOT reuse any example values. ALSO cover, where VALID for the task: (a) the "
    "EDGE/BOUNDARY cases it implies (empty, zero, negative, single element, min/max); (b) at least "
    "one input in a DIFFERENT parameter regime than the visible examples — a different size, scale, "
    "range, or unit; and (c) a probe of the INPUT CONTRACT: feed an input a fragile unit/index/type "
    "assumption would get wrong (e.g. an angle where radians-vs-degrees matters, an off-by-one "
    "index, an int vs float) and assert the correct result. Read every value you get from the "
    "solution or `ref.*` by its TRUE return structure (type/shape/arity/field order), never as a "
    "different shape, and assert that structure so a return-contract mismatch is caught, not silently "
    "passed. Get every EXPECTED value by COMPUTING "
    "it (the reference oracle when provided, else a property) — NEVER a guessed literal. Only assert a "
    "property the SPECIFIC system genuinely has: do NOT assert an idealised behaviour (conservation, "
    "monotonicity, periodicity, boundedness, symmetry) that a term in THIS task breaks (a dissipative / "
    "driving / open / injecting / asymmetric / random term) — check what the task's own rules imply, "
    "not a template from a similar problem. Compare "
    "with tolerances DERIVED FROM THE PROBLEM (math.isclose / numpy.allclose) — a few STANDARD "
    "ERRORS for a stochastic estimate (atol ~= k * stdev / sqrt(N)), the method's error for an "
    "iterative/numerical result, tight only for an exact closed-form value — never an arbitrary "
    "constant that a correct result would trip. Call the SOLUTION's functions directly, and "
    "`assert`. Their purpose is to catch code that is right on the demo value but special-cased, "
    "hardcoded, or made a fragile assumption that breaks on other valid inputs. "
    "Do NOT define the solution or a runner. Output ONLY the Python test functions."
)

_INVARIANTS_SYSTEM = (
    "You write SPEC-DERIVED checks: functions named test_invariant_<name>() that verify the result "
    "against FACTS STATED IN THE REQUEST and properties that must hold for ANY correct "
    "implementation — NOT merely against a same-model reference (a shared wrong assumption could make "
    "both agree). FIRST, write one check PER EXPLICITLY REQUESTED OUTPUT that confirms the reported "
    "value is the EXACT quantity the request names, taken at the EXACT point/index and in the EXACT "
    "units/convention stated — compute the expected value INDEPENDENTLY from the spec, not from the "
    "candidate. Concretely: an 'initial'/'starting' value is the value at the START (t=0 / step 0 / "
    "index 0), a 'final' value at the END; a count is checked for off-by-one; a labelled quantity "
    "must equal that quantity (not a neighbour). A value that is internally consistent but taken at "
    "the wrong point or under a different definition than the request stated is WRONG. Cover any "
    "INTERMEDIATE or COMPONENT value the request asks to report by reading the REAL intermediate the "
    "solution EXPOSES (its function's return) and checking THAT against an independent computation — "
    "never a re-derived or final-stage substitute; an intermediate the solution does not expose is a "
    "contract gap the check must surface. Also assert "
    "each output's STRUCTURE — its type, array shape, and tuple/dict arity — operating on the REAL "
    "returned object (not a re-sliced or re-shaped stand-in); a value of the wrong type or shape, or "
    "one produced by consuming an upstream function's return with the wrong structure, is WRONG even "
    "if a number looks plausible. THEN add "
    "general properties: the given input values and the answer they imply, stated units/conventions, "
    "named constraints, and known identities (e.g. a beamformer's distortionless constraint "
    "w^H d ~= 1; Black-Scholes put-call parity C - P ~= S - K*exp(-rT), price >= 0, monotonic in "
    "volatility). BEFORE asserting ANY such property, CONFIRM THE SPECIFIC SYSTEM IN THIS TASK ACTUALLY "
    "HAS IT — derive it from the task's OWN governing rules/spec, not by analogy to a similar-looking "
    "idealised problem. An idealised property (a CONSERVED / MONOTONIC / PERIODIC / BOUNDED / STATIONARY "
    "/ SYMMETRIC quantity) holds ONLY when the task's own rules imply it; if THIS task includes a "
    "mechanism that BREAKS it — a dissipative / damping / friction term, a driving / forcing / source "
    "term, an open boundary, an injection / removal / birth-death / source-sink process, an explicit "
    "asymmetry, or randomness / noise — that property does NOT hold here and you MUST NOT assert it (no "
    "energy conservation for a DAMPED oscillator, no constant total for an OPEN or driven system, no "
    "monotonicity for a NOISY series, no periodicity for a decaying / chirped signal). For a quantity "
    "the task makes CHANGE, check the change the task's rule PREDICTS, never constancy. "
    "For any result defined only UP TO scaling, sign, ordering, phase, basis, or "
    "representation (an eigenvector/singular vector up to sign or scale, cluster labels up to "
    "permutation, a factorization up to ordering, a basis up to rotation, a vector up to "
    "normalization), do NOT assert one specific representation — assert the DEFINING PROPERTIES every "
    "valid answer must satisfy (e.g. it solves A v = lambda v with unit norm; the clustering induces "
    "the same partition up to relabelling; the factors multiply back to the input). "
    "Use RANDOM inputs where a property is general (use random / numpy.random; do NOT "
    "seed — the harness seeds globally), and the request's own values where the spec pins an answer. "
    "If the request SPECIFIES a METHOD / operation / formula, an output computed a DIFFERENT way is "
    "WRONG: where you can, compute the expected via the SPECIFIED method and assert the candidate "
    "matches it. "
    "ALSO add SANITY checks that catch ASSUMPTION-LEVEL errors a same-assumptions test misses: (i) UNIT "
    "CONSISTENCY / SCALE — the result's units and scale match the spec (a dropped or wrong conversion "
    "is a failure); (ii) ORDER OF MAGNITUDE / PLAUSIBILITY — the result sits in a sensible range with "
    "the correct sign, not off by a large factor; (iii) KNOWN LIMITING CASES — correct behaviour at a "
    "boundary (zero / empty / extreme) or against a known reference value. "
    "Use tolerances DERIVED FROM THE PROBLEM (a few standard errors for a stochastic quantity, the "
    "method's error for a numerical one, tight only for an exact value — never an arbitrary fixed "
    "threshold a correct result would trip), call the SOLUTION's functions directly, and `assert`. Do "
    "NOT define the solution or a runner. Output ONLY the Python test functions."
)

_DEFINITION_SYSTEM = (
    "You write DEFINITION-MATCH checks: exactly ONE function test_definition_<name>() PER explicitly "
    "requested output — INCLUDING any INTERMEDIATE or COMPONENT value the request asks to report (an "
    "intermediate signal or its envelope, a sub-result, a per-stage metric). Each asserts that the "
    "value the solution REPORTS for that output is the EXACT thing the user asked for — the right "
    "QUANTITY, at the right POINT/TIME, with the right AGGREGATION, in the right UNITS/CONVENTION — "
    "and not a related-but-different quantity. For an intermediate value, OBTAIN it from the function "
    "that must EXPOSE it (its real return) and verify THAT against your independent computation — if "
    "the solution does not expose the intermediate (its functions return only the final result), the "
    "check cannot get the real value and MUST fail, so the gap is fixed by exposing it rather than "
    "papered over with a fabricated stand-in. "
    "FIRST assert the value's STRUCTURE matches the spec — the correct type (scalar vs array vs tuple "
    "vs dict), shape/dimensions, tuple/dict arity and field names, and units — reading the ACTUAL "
    "object the solution returns by its true contract; do NOT slice, unpack, or reshape it into a "
    "wrong-shaped stand-in before checking, or the check would validate the wrong object. A consumer "
    "that read some function's return with the wrong shape/type yields a structurally wrong output "
    "and MUST fail here. "
    "Re-read the REQUEST and derive the expected value INDEPENDENTLY from the user's own wording, "
    "computing it by a DIFFERENT route than the solution uses. Do NOT trust or call any reference "
    "implementation here — a same-model reference can encode the SAME wrong definition, so 'the code "
    "matches the reference' must NOT be how you judge this. You MUST catch each of: a related-but-"
    "different quantity (e.g. mean reported for median, sum for average, radius for diameter, "
    "variance for std-dev); a wrong AGGREGATION (sum vs mean vs max vs last); a wrong REFERENCE POINT "
    "(initial vs after-the-first-step, final vs penultimate, inclusive vs exclusive endpoint, "
    "off-by-one count); a wrong UNIT/CONVENTION (degrees vs radians, fraction vs percent, 0- vs "
    "1-based); and a LABEL that claims one quantity while the logic returns another. If the request "
    "SPECIFIES a METHOD / operation / formula for an output, compute the expected value USING that "
    "specified method/formula — so a candidate that used a DIFFERENT method (which can give a different "
    "value, e.g. a causal filter vs a convolution) FAILS this check. For each output, "
    "build concrete spec inputs (the request's own values when it gives them), call the SOLUTION's "
    "function(s), compute the spec-correct expected value YOURSELF, and `assert` they match (use "
    "math.isclose / numpy.allclose for floats, == for exact). Do NOT define the solution or a runner. "
    "Output ONLY the Python test functions."
)

_CRITIC_SYSTEM = (
    "You are the TEST-CRITIC: a reviewer that audits the GENERATED TESTS/CHECKS for a task BEFORE any "
    "of them is allowed to judge a candidate solution. You do NOT write solution code and you do NOT "
    "judge the candidate — you judge and REPAIR the TESTS so an INVALID test can never fail correct "
    "code. Audit EACH test; a test is INVALID if it would fail a CORRECT solution for any of:\n"
    "1. GUESSED EXPECTED VALUE — the expected value was imagined, not derived from the spec or by "
    "executing the independent REFERENCE provided. Replace it with the value computed from the spec or "
    "the reference (call ref.* at runtime; do not paste a literal you cannot justify).\n"
    "2. WRONG / HARDCODED TOLERANCE — a fixed constant where the tolerance must come from the MATH: a "
    "stochastic estimate needs a few standard errors (~ k*stdev/sqrt(N), from the sample size/variance), "
    "a numerical result needs the method's error. Replace the constant with a tolerance derived from the "
    "problem.\n"
    "3. EXACT-MATCH ON A NON-UNIQUE QUANTITY — asserts exact equality on a result defined only up to "
    "scaling / sign / ordering / phase / basis / representation, or produced by an underdetermined "
    "procedure. Replace it with the DEFINING PROPERTIES every valid answer must satisfy (e.g. A v = "
    "lambda v with unit norm; the same partition up to relabelling; the factors multiply back to the "
    "input).\n"
    "4. WRONG OPERATIONAL DEFINITION — encodes a naive PROXY for a named property instead of its TRUE "
    "definition (e.g. 'flat / uniform' as every-element-equals-the-mean rather than no-dominant-"
    "component; 'sorted' as strictly-monotonic when equal values are allowed). Replace the proxy with "
    "the property's real definition.\n"
    "5. REQUIREMENT NOT IMPLIED BY THE TASK — asserts a property/invariant THIS task does not have or "
    "deliberately BREAKS (a conservation / preservation / monotonicity / periodicity assertion on a "
    "task whose own rules change that quantity — a dissipative, driving, open, injecting, asymmetric, "
    "or random term). The property must follow from THIS task's spec and the SPECIFIC system, not a "
    "generic template. DROP such a check, or replace it with the change the task's rule actually "
    "predicts.\n"
    "6. WRONG QUANTITY / ENTITY — tests a different value than the request asked for (a neighbour, a "
    "component, the wrong point / index / units). Re-target it at the EXACT requested quantity.\n"
    "HARD RULES: keep every VALID test VERBATIM. Do NOT weaken a valid check to make code pass, and do "
    "NOT make any check lenient enough to pass a genuinely WRONG result — repairing a test means making "
    "it test the TRUE requirement, never making it loose. Keep EXACT checks STRICT for genuinely UNIQUE "
    "quantities. Preserve each test's calling convention and the solution's function names/signatures. "
    "Output the REPAIRED suite — valid tests unchanged, invalid tests rewritten to the true requirement, "
    "requirement-not-implied checks omitted — as plain Python test_* functions and NOTHING else. Do NOT "
    "define the solution or a runner."
)

_DIAGNOSE_SYSTEM = (
    "You are the DIAGNOSIS agent — a debugging engineer/scientist doing ROOT-CAUSE analysis on a "
    "candidate that FAILED one or more checks, in WHATEVER domain the task is (physics, biology, "
    "chemistry, finance, ecology, signal/audio, ...). You are a SEPARATE role from the generation "
    "agent: you do NOT write the final code — you DIAGNOSE the MECHANISM and hand the generation agent "
    "a precise FIX DIRECTIVE to apply. Do NOT propose a random change — reason from the SYMPTOM (which "
    "check failed, the traceback, the numbers) to the SPECIFIC place in the code that is wrong. The "
    "mapping is by the SHAPE of the failure and is "
    "DOMAIN-INDEPENDENT — the same shape points to the same kind of cause whether the quantity is "
    "energy, total population, total probability, mass, momentum, money, a total count, or signal "
    "power. Map symptom -> cause:\n"
    "1. A quantity that SHOULD be CONSERVED or INVARIANT DRIFTS or is not conserved (any conserved / "
    "bounded quantity the task or its domain implies — energy, total probability, total population, "
    "mass, momentum, money, a norm, a total count, signal power) AND the task's system GENUINELY "
    "conserves it — if THIS task includes a term that BREAKS conservation (a dissipative / driving / "
    "open / injection-removal / asymmetric / random term) the quantity SHOULD change, the drift is "
    "CORRECT and you must NOT diagnose it (check the change the task's rule predicts instead) -> the "
    "UPDATE / ITERATION / "
    "INTEGRATION RULE is the likely cause: a wrong update formula, wrong sign, wrong coefficient, or a "
    "method that does not preserve the quantity. Name the rule and the fix (use a conservation- or "
    "structure-preserving update) — NOT a looser tolerance.\n"
    "2. A result has the WRONG SIGN or WRONG DIRECTION (something that must grow shrinks or vice-versa, "
    "a quantity that must stay non-negative goes negative, a trend or movement opposite to what the "
    "inputs dictate) -> a SIGN or term-ORDERING error in the GOVERNING EQUATION / RULE. Point to the "
    "offending term and correct it.\n"
    "3. Values BLOW UP or go NaN / inf -> NUMERICAL INSTABILITY (the step or rate is too large for the "
    "scheme, a missing stability or normalisation factor) or a divide-by-zero. Fix the scheme or step "
    "size — do NOT just clamp the output.\n"
    "4. A check passes ONLY because the code FORCES it (renormalising the state every step so an "
    "invariant is trivially satisfied, clamping the output into range, hardcoding the expected value) "
    "-> this is MASKING, not solving. Call it out and require the property to EMERGE from a correct "
    "rule: fix the rule at the SOURCE and REMOVE the forced enforcement. Never accept masking.\n"
    "5. The WRONG ENTITY / WRONG QUANTITY is reported -> an output-selection bug: fix which value is "
    "computed or printed.\n"
    "If a PREVIOUS diagnosis is shown and the same check is STILL failing, that fix did NOT work — "
    "diagnose a DIFFERENT mechanism this time. Output EXACTLY these three labelled lines, plain text, "
    "no code:\n"
    "FAILING CHECK: <name the check that failed; its observed behaviour vs. the expected behaviour>.\n"
    "MECHANISM: <which failure shape above; the exact code location / rule responsible>.\n"
    "FIX DIRECTIVE: <one concrete change for the generation agent — a corrected mechanism at its "
    "source; NEVER a looser tolerance, and NEVER forced renormalisation / clamping / hardcoding (that "
    "is masking) — the property must EMERGE from the corrected rule>."
)

# Task-type-specific guidance appended to the test/invariant generation USER prompts (the system
# prompts above stay fixed) so verification matches the KIND of task. The model DERIVES 2-4
# concrete properties appropriate to the type — a single "expected output" is meaningless for a
# stochastic simulation, so those are checked by invariants instead.
_TASK_TYPE_GUIDANCE = {
    "deterministic": (
        "\n\nTASK TYPE = DETERMINISTIC: a single correct output exists. Assert EXACT expected "
        "outputs on small concrete inputs (use math.isclose / numpy.allclose only for floats, with "
        "a TIGHT tolerance ~1e-9 — correct here because the output is exact), PLUS 2-4 general "
        "properties any correct solution must satisfy (e.g. output length and ordering, idempotence, "
        "boundary/empty cases)."
    ),
    "simulation": (
        "\n\nTASK TYPE = SIMULATION / STOCHASTIC: there is NO single fixed output — do NOT assert "
        "one magic number. DERIVE 2-4 INVARIANTS / PROPERTIES on the REAL output: correct output "
        "shape/type; conservation laws (energy / mass / probability sums); values within physical "
        "or range bounds; expected convergence or a monotonic trend (e.g. a damped system's "
        "amplitude decreases over time); and seeded REPRODUCIBILITY (same seed -> identical "
        "output). SIZE each tolerance from the math, not by guessing: when you check a stochastic "
        "mean/estimate over N samples, a correct run is EXPECTED to differ from the true value by "
        "about one STANDARD ERROR, so allow a few SE (atol ~= k * stdev / sqrt(N), k=3-5); for a "
        "discretized quantity use the discretization error. Never a tiny fixed atol on a noisy "
        "estimate — it would falsely fail a correct simulation."
    ),
    "numeric_algorithm": (
        "\n\nTASK TYPE = NUMERIC ALGORITHM: assert DOMAIN INVARIANTS, not one value. DERIVE 2-4 "
        "mathematical properties that hold for ANY correct implementation, e.g. a beamformer's "
        "distortionless constraint w^H d ~= 1 and output noise power <= input; FFT / Parseval "
        "energy conservation; Black-Scholes put-call parity C - P ~= S - K*exp(-rT), price >= 0 "
        "and monotonic in volatility. Compare with a tolerance SIZED TO THE METHOD'S ERROR (step "
        "size h, iteration/convergence tol), not an arbitrary constant a correct result would trip."
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


def _generate_reference(provider, task: str, requirements: str, task_type: str = "",
                        divergent: bool = False) -> str:
    """(oracle) A clear, correct reference implementation whose EXECUTED outputs become the expected
    values the tests assert against — so 'expected' is computed, never guessed. Returns '' on any
    failure, and the caller falls back to property/legacy tests.

    `divergent=True` requests a SECOND, INDEPENDENT reference used only to DETECT non-unique results:
    where the answer is defined merely up to scaling/sign/ordering/phase/basis/representation, it
    deliberately returns a DIFFERENT-but-valid representation, so comparing the two references reveals
    which exact tests are asserting one arbitrary representation of a non-unique quantity."""
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\n"
            "Write the reference implementation now — the SAME functions the task requires.")
    if divergent:
        user += (
            "\n\nIMPORTANT — this is a SECOND, INDEPENDENT reference used to detect NON-UNIQUE results. "
            "Use a genuinely DIFFERENT method from the most obvious one. Where the result is defined "
            "ONLY up to scaling, sign, ordering, phase, basis, or representation (or comes from an "
            "underdetermined / non-canonical procedure), deliberately RETURN A DIFFERENT BUT EQUALLY "
            "VALID representation — e.g. the opposite-sign eigenvector, a different valid ordering, a "
            "different basis, a differently-but-validly normalized vector. For results that ARE "
            "uniquely determined, return the SAME correct value. Define ONLY the functions.")
    try:
        return _extract_code(_complete(provider, _REFERENCE_SYSTEM, user, GEN_MAX_TOKENS))
    except Exception:
        return ""


_OUTPUT_INTENT = re.compile(
    r"\b(print|show|display|report|return|output|result|compute|calculate|simulate|find|"
    r"estimate|measure|benchmark|value|price|how\s+many|what\s+is)\b", re.I)


def _wants_output(task: str) -> bool:
    """True if the request asks to print/show/return/report a RESULT — then the final answer must
    include the real captured stdout from running the solution, not just code."""
    return bool(_OUTPUT_INTENT.search(task or ""))


def _generate_demo_driver(provider, task: str, requirements: str, solution_code: str) -> str:
    """A short snippet that calls the finished solution on representative inputs and prints the
    real results, so the user sees actual values. '' on failure (no demo run)."""
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\nSOLUTION (already defined — call it, "
            f"do not redefine):\n```python\n{solution_code[:3000]}\n```\n\n"
            "Write the driver snippet now: call the solution and print the requested result(s). "
            "Summarize any large array/matrix/dataset (shape + a few values) instead of dumping it, "
            "and print the requested values in a clear labelled block LAST.")
    try:
        return _extract_code(_complete(provider, _DRIVER_SYSTEM, user, REFLECT_MAX_TOKENS))
    except Exception:
        return ""


# ----------------------------------------------------------------------
# Completeness + execution gates: every requested output must appear in real stdout.
# ----------------------------------------------------------------------
_DELIVERABLES_SYSTEM = (
    "You extract the explicit DELIVERABLES of a coding request: the distinct things the finished "
    "program must OUTPUT when it runs — each value to print/return, each comparison, each property "
    "to report, INCLUDING any INTERMEDIATE or COMPONENT value it asks for (an intermediate signal or "
    "envelope, a sub-result, a per-stage metric), not only the final result. Output a short PLAIN "
    "LIST, one deliverable per line, each a 1-4 word lowercase label "
    "naming the quantity by its EXACT stated meaning and aggregation (e.g. 'sum of coefficients' — NOT "
    "'peak'; 'rms of filtered signal' — NOT 'per-component rms'; 'period', 'kinetic energy', 'put-call "
    "parity'). Capture the precise quantity the request NAMES, not a related-but-different one. No "
    "numbers, no code, no prose, no bullets — only the labels, one per line. If the request asks for "
    "nothing to be output, return an empty response."
)

_DELIVERABLE_STOP = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "value", "values",
                     "result", "results", "output", "each", "every", "all", "its", "with", "is"}
_WORD = re.compile(r"[a-z0-9]+")


def _parse_deliverables(text: str) -> List[str]:
    """Parse the deliverables-extraction output into a clean, deduped list of short labels."""
    items: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*•0123456789.) ").strip().strip("`").strip().lower()
        if not line or len(line) > 40 or line.startswith(("def ", "import ", "#", "print(")):
            continue
        if line not in items:
            items.append(line)
    return items[:12]


def _extract_deliverables(provider, task: str, requirements: str) -> List[str]:
    """LLM-extracted checklist of the explicit outputs the request asks for (completeness gate).
    Best-effort: returns [] on any failure, so the gate passes vacuously rather than blocking."""
    try:
        user = (f"REQUEST:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\n"
                "List the deliverables now, one short label per line.")
        return _parse_deliverables(_complete(provider, _DELIVERABLES_SYSTEM, user, REFLECT_MAX_TOKENS))
    except Exception:
        return []


def _check_completeness(deliverables: List[str], stdout: str) -> List[str]:
    """Return the deliverables NOT evidenced in `stdout`. A deliverable is present when all of its
    significant tokens (stop-words removed) appear in the output, case-insensitively — so a label
    like 'kinetic energy' matches a line printing 'Kinetic energy: 5.0'. Empty list -> none missing."""
    out = (stdout or "").lower()
    out_tokens = set(_WORD.findall(out))
    missing: List[str] = []
    for d in deliverables or []:
        toks = [t for t in _WORD.findall(d.lower()) if len(t) > 2 and t not in _DELIVERABLE_STOP]
        if not toks:                                    # all-short label -> require the raw substring
            if d.lower().strip() and d.lower().strip() not in out:
                missing.append(d)
        elif not all(t in out_tokens for t in toks):
            missing.append(d)
    return missing


def _apply_output_gates(verdict: Dict[str, Any], *, wants_output: bool, output: str,
                        missing: List[str]) -> Dict[str, Any]:
    """EXECUTION + COMPLETENESS gates on a candidate that already cleared the visible + held-out
    (robustness/spec) gates. Downgrades verified/done with honest reasons when the task asked for
    output but produced none, or a requested deliverable is missing from the real stdout. Never
    resurrects a verdict that already failed an earlier gate."""
    if not verdict.get("verified"):
        return verdict
    reasons: List[str] = []
    if wants_output and not (output or "").strip():
        reasons.append("execution: the request asks for output but the solution produced no real stdout")
    if missing:
        reasons.append("completeness: requested output(s) missing from stdout: " + ", ".join(missing))
    if reasons:
        verdict["verified"] = False
        verdict["done"] = False
        verdict["gate_fail"] = "; ".join(reasons)
        verdict["feedback"] = (
            "Your solution passed the tests but FAILED a delivery gate — " + "; ".join(reasons)
            + ". Defining the functions is not enough: add a runnable `if __name__ == \"__main__\":` "
            "entry point that CALLS your functions on the task's inputs and PRINTS every requested "
            "value with a clear label, so running the file produces real output. Do NOT dump large "
            "arrays/matrices/datasets in full (that buries the answer) — print a compact summary for "
            "those, and print every requested value in a clear FINAL labelled block at the very END.")
    return verdict


def _capture_and_check(code: str, deliverables: List[str]):
    """Run the finished SOLUTION ITSELF as a script (its `if __name__ == "__main__":` block runs and
    PRINTS the requested values) and capture its REAL stdout — NOT a separately-generated driver, so a
    solution that defines functions but never calls/prints them produces empty output and FAILS here.
    Returns (output, missing_deliverables)."""
    output = ""
    try:
        dres = run_python_auto(code)
        if dres.ok and (dres.stdout or "").strip():
            # Head+tail clip: a value printed LAST (after a big intermediate dump) must survive so the
            # completeness gate can see it and the user can read it.
            output = clip_keep_ends((dres.stdout or "").strip(), DEMO_OUTPUT_CAP)
    except Exception:                                   # noqa: BLE001 - capture failures -> empty output
        output = ""
    missing = _check_completeness(deliverables, output) if deliverables else []
    return output, missing


def _latest_revalidated_delivery(attempts: List[Attempt], deliverables: List[str],
                                 gate_output: bool) -> Optional[Attempt]:
    """RE-VALIDATE AGAINST THE LATEST ATTEMPT before settling for a STALE 'partial'. The attempt the
    loop selected (highest-scoring clean one) can be an EARLIER attempt that was demoted for 'missing
    output', while the MOST RECENT attempt actually prints every deliverable — so the failure was
    judged against a stale state, not the latest code. Scan attempts MOST-RECENT first for the first
    LEGITIMATE one that already cleared the visible + held-out gates (so only the delivery/completeness
    gate could have demoted it); re-run THAT exact code ONCE and re-check the deliverables against
    FRESH stdout. If it now produces every deliverable, return it with the delivery verdict refreshed
    to verified. Returns None when nothing should be upgraded — the latest good attempt is already
    verified, has a GENUINE non-delivery failure (held-out/visible/cheating/off-topic), or its fresh
    run is still incomplete. Bounded to ONE re-run. Re-validates by RE-RUNNING real code: it can only
    retire the one gate (delivery) the live code provably no longer fails — it never resurrects a
    held-out/visible/cheating/off-topic failure (those are hard exclusions)."""
    if not gate_output:
        return None
    for att in reversed(attempts):
        v = att.verdict
        if v.get("cheating") or v.get("relevant") is False:
            continue                       # skip gaming / off-topic; judge the latest LEGITIMATE one
        if v.get("verified"):
            return None                    # the latest good attempt is already verified -> nothing stale
        total, passed = int(v.get("total", 0)), int(v.get("passed", 0))
        htot, hpass = int(v.get("hidden_total", 0)), int(v.get("hidden_passed", 0))
        visible_ok = total > 0 and passed >= total
        heldout_ok = htot == 0 or hpass >= htot
        if v.get("hidden_fail") or not (visible_ok and heldout_ok):
            return None                    # a GENUINE non-delivery failure -> the honest partial stands
        out, missing = _capture_and_check(att.code, deliverables)   # re-run THIS code, FRESH stdout
        if (out or "").strip() and not missing:
            nv = dict(v)
            nv["verified"], nv["done"], nv["demo_output"] = True, True, out
            nv["gate_fail"] = ""           # the delivery gate the live code provably no longer fails
            nv["diagnosis"] = ""           # the stale 'missing output' diagnosis no longer applies
            return Attempt(att.iteration, att.code, att.result, nv)
        return None                        # latest eligible attempt genuinely incomplete -> partial
    return None


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
                       temperature: float = 0.2, variant: str = "", frozen: str = "",
                       wants_output: bool = False) -> str:
    """(c) Write modular solution code so the provided tests pass. The tests are appended by the
    runner, not by the model. `memory_summary` carries the cross-attempt 'avoid these' notes.
    `frozen` (a _freeze_clause) tells the model to KEEP the already-passing parts unchanged and
    revise only the failing ones. `temperature`/`variant` diversify parallel best-of-N candidates
    WITHOUT implying failure — only `feedback` (real diagnostics from a prior round) signals
    "fix what failed"."""
    parts = [f"TASK:\n{task}", f"\nREQUIREMENTS:\n{requirements}",
             "\nYour solution MUST define the functions these tests call so they pass. Solve the "
             "GENERAL problem — do NOT hardcode the expected outputs, special-case these specific "
             "inputs, read the tests, or fake the functions; your code is also checked on unseen "
             f"random inputs. Do NOT include the tests or a test runner:\n```python\n{tests}\n```"]
    if reference:
        parts.append(f"\nREFERENCE implementations (adapt the approach, do NOT copy):\n{reference[:3000]}")
    if last_code:
        parts.append(f"\nYOUR PREVIOUS SOLUTION (keep what's correct, fix what's not):\n"
                     f"```python\n{last_code}\n```")
    if frozen:
        parts.append("\n" + frozen)
    if memory_summary:
        parts.append("\nAVOID repeating these already-failed or REJECTED approaches:\n" + memory_summary)
    if feedback:
        parts.append("\nThe tests FAILED last time. If the DIAGNOSIS AGENT's FIX DIRECTIVE is given "
                     "below, APPLY that mechanism fix at its source (do not merely loosen tolerances or "
                     "mask the symptom by renormalising/clamping); then address the specific PASS/FAIL "
                     f"lines and tracebacks (do not rewrite from scratch):\n{feedback[:3000]}")
    if wants_output:
        parts.append("\nThis task asks for OUTPUT. Your file MUST be a RUNNABLE PROGRAM: add an "
                     "`if __name__ == \"__main__\":` block that calls your functions on the task's "
                     "inputs and PRINTS every requested value with a clear text label (a compact "
                     "summary — shape/length + a few values — for any large array/matrix/dataset, "
                     "never a full dump). Defining the functions WITHOUT a call site that prints is "
                     "INCOMPLETE — the file will be run and its real stdout checked for every value.")
    if variant:
        parts.append("\n" + variant)
    parts.append("\nWrite the solution code now (define the functions; for an output task also add "
                 "the runnable `__main__` that prints the results).")
    return _extract_code(_complete(provider, _GEN_SYSTEM, "\n".join(parts), GEN_MAX_TOKENS,
                                   temperature=temperature))


def _diagnose_failure(provider, task: str, requirements: str, code: str, symptom: str,
                      failing: List[str], prev_diagnosis: str = "", repeated: bool = False) -> str:
    """The DIAGNOSIS agent: a SEPARATE role/prompt from generation. Root-causes the most recent failure
    by mapping the SYMPTOM (failing checks + traceback) to the likely MECHANISM and code location, and
    emits a STRUCTURED directive (FAILING CHECK / MECHANISM / FIX DIRECTIVE) that the generation agent
    then applies — so the next rewrite targets the cause, not a blind mutation. Runs ONCE per failed
    round (never per candidate), on the same resilient provider, so it adds no unbounded fan-out.
    Returns '' on any failure (fail-open) — a diagnosis hiccup degrades to the raw traceback feedback
    rather than breaking the loop. When the same check keeps failing, asks for a DIFFERENT mechanism."""
    if not root_cause_enabled() or not (code or "").strip() or not (symptom or "").strip():
        return ""
    fail_list = ", ".join(failing[:8]) if failing else "(see the traceback)"
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\n"
            f"FAILING CHECK(S): {fail_list}\n\n"
            f"SYMPTOM (the runner's output / traceback):\n{(symptom or '')[:2000]}\n\n"
            f"THE CODE THAT FAILED:\n```python\n{(code or '')[:3000]}\n```")
    if prev_diagnosis:
        tag = (" The SAME check is STILL failing, so that fix did NOT work — diagnose a DIFFERENT "
               "mechanism." if repeated else "")
        user += f"\n\nPREVIOUS DIAGNOSIS (last round):\n{prev_diagnosis[:800]}{tag}"
    user += "\n\nDiagnose the root cause now."
    try:
        return (_complete(provider, _DIAGNOSE_SYSTEM, user, GEN_MAX_TOKENS) or "").strip()[:1200]
    except Exception:
        return ""


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


def _generate_definition_checks(provider, task: str, requirements: str, strict: bool = False,
                                task_type: str = "") -> str:
    """DEFINITION-MATCH gate: one held-out check per requested output asserting the REPORTED value is
    the exact quantity the user asked for (quantity / point / aggregation / units), with the expected
    computed INDEPENDENTLY from the request — never via the candidate or its reference oracle. Catches
    'right logic, wrong reported answer'. Returns '' on failure (the held-out then degrades to hidden/
    invariant checks)."""
    extra = (" Be especially strict: probe the most likely wrong-quantity / wrong-aggregation / "
             "wrong-point confusions for these outputs.") if strict else ""
    user = (f"TASK (the user's exact request):\n{task}\n\nREQUIREMENTS (each deliverable's exact "
            f"definition):\n{requirements}" + _task_type_hint(task_type) +
            "\n\nWrite the test_definition_* functions now — one per requested output, each asserting "
            "the solution's reported value matches that output's EXACT stated definition, computed "
            "independently from the request (do not call any reference)." + extra)
    try:
        return _extract_code(_complete(provider, _DEFINITION_SYSTEM, user, GEN_MAX_TOKENS))
    except Exception:
        return ""


def _critique_tests(provider, task: str, requirements: str, tests: str, oracle: str = "",
                    kind: str = "tests") -> str:
    """TEST-CRITIC (a SEPARATE role from generation and diagnosis): audit the generated suite and return
    a REPAIRED version where every INVALID test — a GUESSED expected value, a HARDCODED tolerance where
    the math implies one, an EXACT-match on a NON-UNIQUE quantity, a WRONG operational definition, a
    requirement NOT IMPLIED by the task, or the WRONG quantity — is rewritten to test the TRUE
    requirement (or dropped, for a requirement the task does not imply), while VALID tests are kept
    verbatim. ONE bounded LLM call per suite (never per candidate); the repaired suite is still
    validated against the reference downstream. Fail-OPEN: returns the ORIGINAL suite on any error,
    empty output, or output defining no test function, so a hiccup degrades to the execution quarantine
    alone."""
    if not test_critic_enabled() or not (tests or "").strip():
        return tests
    ref = (oracle or "").strip()
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\n"
            "INDEPENDENT REFERENCE implementation of THIS task (use it to DERIVE expected values and to "
            "sanity-check — a test the correct reference itself fails is invalid by construction):\n"
            f"{ref[:3000] if ref else '(no reference available — derive expected values from the spec)'}"
            f"\n\nGENERATED {kind} to audit and REPAIR:\n```python\n{tests}\n```\n\n"
            "Audit each test against the six invalid-test reasons and output the REPAIRED suite now.")
    try:
        repaired = _extract_code(_complete(provider, _CRITIC_SYSTEM, user, GEN_MAX_TOKENS))
    except Exception:                       # noqa: BLE001 - critic is best-effort; never break the run
        return tests
    # Accept only a non-trivial repaired suite that still defines test functions (else fail open).
    if not (repaired or "").strip() or _count_tests(repaired) == 0:
        return tests
    return repaired


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


# ----------------------------------------------------------------------
# Oracle test-validation: a test the KNOWN-CORRECT reference itself fails is invalid (guessed
# expected, too-tight tolerance, or wrong definition) and must never be allowed to fail a candidate.
# ----------------------------------------------------------------------
def _test_results(stdout: str) -> Dict[str, bool]:
    """Parse the per-test 'TEST <name> PASS|FAIL' lines the runner prints into {name: passed}. Empty
    when those lines are absent (e.g. the script crashed before the runner)."""
    return {m.group(1): (m.group(2) == "PASS")
            for m in re.finditer(r"^TEST\s+(\w+)\s+(PASS|FAIL)\s*$", stdout or "", re.M)}


def _passing_names(stdout: str) -> List[str]:
    """Names of the checks that PASSED in this run — the verified-correct parts to FREEZE."""
    return [n for n, ok in _test_results(stdout).items() if ok]


def _failing_names(stdout: str) -> List[str]:
    """Names of the checks that FAILED — the only parts the next attempt should revise."""
    return [n for n, ok in _test_results(stdout).items() if not ok]


def _genuine_failing(stdout: str, quarantine: set) -> List[str]:
    """Failing check NAMES with the QUARANTINE removed, so the reported failures AGREE with the
    quarantine-adjusted pass/total. A check the reference oracle ITSELF fails (quarantined: guessed
    value, too-tight tolerance, exact-match-on-non-unique, or wrong definition) is INVALID — a candidate
    'failing' it is not a real failure, so it must never be reported, fed to the diagnosis, or logged as
    a failure pattern, even though it shows as FAIL in the raw stdout."""
    return [n for n in _failing_names(stdout) if n not in (quarantine or set())]


def _freeze_clause(passing: List[str], regressed: List[str] | None = None) -> str:
    """Refinement instruction that FREEZES verified-correct parts: the next attempt keeps the code
    satisfying the already-passing checks UNCHANGED and revises ONLY what the failing checks need.
    `regressed` (checks that were green before but a later attempt re-broke) are called out so the
    agent RESTORES them. Empty string when there is nothing yet proven correct to preserve."""
    if not passing and not regressed:
        return ""
    lines: List[str] = []
    if passing:
        shown = ", ".join(sorted(passing)[:20])
        lines.append(
            "ALREADY-PASSING checks (your previous solution is CORRECT for these — KEEP the code that "
            f"satisfies them UNCHANGED, do NOT rewrite working functions): {shown}.")
    if regressed:
        lines.append(
            "You RE-BROKE these previously-passing checks last attempt — RESTORE them while keeping "
            f"the rest: {', '.join(sorted(regressed)[:20])}.")
    lines.append("Revise ONLY the parts responsible for the FAILING checks below; leave everything "
                 "that already passes exactly as it is.")
    return "\n".join(lines)


def _invalid_tests(oracle_code: str, tests_code: str, footer: str = _TEST_FOOTER) -> set:
    """THE UNIVERSAL TEST-ADMISSION GATE: run the KNOWN-CORRECT reference ORACLE through the generated
    tests and return the names of the tests it FAILS. A correct reference failing a test means the TEST
    is wrong — by CONSTRUCTION and regardless of WHY (a guessed expected value, a tolerance too tight for
    the method, an exact-match on a non-unique quantity, a wrong definition, a property the task does not
    imply, a malformed/erroring test, or ANY cause not enumerated here) — so that test is quarantined and
    never allowed to fail a candidate. This is ONE rule, not a list of recognised defects. The oracle is
    BOTH solution and reference here, so any `ref.*`-derived expected trivially agrees; what surfaces is
    the tests (properties / definitions / guessed literals / non-task assertions / too-tight tolerances)
    that even correct code cannot satisfy. Fail-OPEN: returns an empty set with no oracle, or when no
    per-test lines parse (a crash), so a hiccup never silently drops genuine tests."""
    if not (oracle_code or "").strip() or not (tests_code or "").strip():
        return set()
    try:
        result, _passed, _total = _run_against_tests(oracle_code, tests_code, footer,
                                                     reference_src=oracle_code)
        results = _test_results(result.stdout or "")
        if not results:
            return set()
        failing = {name for name, ok in results.items() if not ok}
        # A reference that fails its OWN ENTIRE suite is unreliable (e.g. wrong function names -> every
        # test errors). Don't trust it to quarantine anything — fail OPEN rather than quarantine all
        # (which would otherwise leave each candidate judged on the very tests proven invalid).
        return set() if len(failing) == len(results) else failing
    except Exception:                       # noqa: BLE001 - validation is best-effort; fail open
        return set()


def _heldout_quarantine(oracle_code: str, heldout_code: str, seeds: Optional[int] = None) -> set:
    """THE UNIVERSAL GATE for the held-out suite: a generated check may judge a candidate ONLY IF the
    known-correct reference itself PASSES it. EVERY held-out check — hidden, invariant, AND definition —
    that the correct reference FAILS is quarantined, by construction and regardless of WHY (see
    _invalid_tests): if the correct answer cannot satisfy the test, the TEST is wrong, not the code.
    (Definition checks still compute their expected value INDEPENDENTLY from the spec — only their
    ADMISSION is reference-gated; the test-critic remains the complementary semantic layer for the rare
    case of a reference that shares a wrong definition.) Validated across the SAME random seeds
    `_verify_heldout` judges candidates on (UNION): a stochastic check the correct reference fails on ANY
    of those seeds is flawed, so quarantining it covers every seed — not just the first. Empty unless
    test-validation is on and both an oracle and held-out code are present."""
    if not (test_validation_enabled() and (oracle_code or "").strip() and (heldout_code or "").strip()):
        return set()
    n = max(1, seeds if seeds is not None else verify_seeds())
    invalid: set = set()
    for s in range(n):
        invalid |= _invalid_tests(oracle_code, heldout_code, _seeded_footer(1000 + s))
    return invalid


def _nonunique_exact_tests(provider, task: str, requirements: str, task_type: str,
                           oracle_code: str, all_tests: str, seeds: int) -> set:
    """Differential NON-UNIQUE detection (extends the quarantine). Many correct answers are defined
    only up to scaling / sign / ordering / phase / basis / representation, or come from an
    underdetermined procedure. A test asserting EXACT equality to ONE such representation passes the
    single oracle (it agrees with itself) yet FAILS a different valid solution — so it is invalid.

    We prove this by EXECUTION, not guesswork: generate an INDEPENDENT cross-reference that returns a
    DIFFERENT valid representation for non-unique results, then — ONLY if that cross-reference satisfies
    every PROPERTY check (`test_invariant_*` / `test_definition_*`), confirming it is a genuinely
    CORRECT alternative — any EXACT test it FAILS that the primary oracle PASSES is quarantined. A
    uniquely-determined quantity makes the cross-reference return the SAME value, so its exact tests are
    never quarantined. Fail-OPEN (empty set) when disabled, no cross-reference, the cross-reference
    can't be validated, or on any error — a hiccup must never drop a genuine exact test."""
    if not nonunique_validation_enabled() or not (oracle_code or "").strip() or not (all_tests or "").strip():
        return set()
    try:
        alt = _generate_reference(provider, task, requirements, task_type, divergent=True)
        if not (alt or "").strip():
            return set()
        nonunique: set = set()
        for s in range(max(1, seeds)):
            footer = _seeded_footer(2000 + s)
            oracle_fail = _invalid_tests(oracle_code, all_tests, footer)   # tests even the oracle can't satisfy
            res, _p, _t = _run_against_tests(alt, all_tests, footer, reference_src=oracle_code)
            results = _test_results(res.stdout or "")
            if not results:
                return set()                                              # cross-reference crashed -> trust nothing
            props = {x: ok for x, ok in results.items()
                     if x.startswith(("test_invariant", "test_definition"))}
            if not props or not all(props.values()):
                return set()       # cross-reference is NOT a validated correct alternative -> fail open
            nonunique |= {x for x, ok in results.items()
                          if not ok and x not in oracle_fail
                          and not x.startswith(("test_invariant", "test_definition"))}
        return nonunique
    except Exception:              # noqa: BLE001 - best-effort; never drop a genuine test on a hiccup
        return set()


def _valid_counts(stdout: str, quarantine: set):
    """Recompute (passed, total) over only the NON-quarantined tests, from the per-test PASS/FAIL
    lines. Returns None when no per-test lines parse (script crashed before the runner) or every test
    is quarantined — the caller then keeps the original tally rather than inventing one."""
    results = _test_results(stdout)
    if not results:
        return None
    valid = {n: ok for n, ok in results.items() if n not in (quarantine or set())}
    if not valid:
        return None
    return sum(1 for ok in valid.values() if ok), len(valid)


def _verify_heldout(solution_code: str, heldout_code: str, seeds: int, reference_src: str = "",
                    quarantine: Optional[set] = None):
    """(C4) Run solution + held-out (hidden + invariant) tests once per random seed, judging the
    fresh inputs against the SAME reference oracle. Returns (ok_all_seeds, passed, total,
    last_result); ok only if EVERY seed fully passes. No held-out -> (True, 0, 0, None). Held-out
    tests the reference oracle itself fails (`quarantine`) are excluded from the tally, so an invalid
    held-out test never falsely fails a correct candidate."""
    total0 = _count_tests(heldout_code)
    if not total0:
        return True, 0, 0, None
    quarantine = quarantine or set()
    last, last_total = None, total0
    for s in range(max(1, seeds)):
        result, passed, total = _run_against_tests(
            solution_code, heldout_code, _seeded_footer(1000 + s), reference_src=reference_src)
        last = result
        if quarantine:                       # judge on valid (oracle-passing) held-out tests only
            vc = _valid_counts(result.stdout or "", quarantine)
            if vc is not None:
                passed, total = vc
        last_total = total or total0
        if total == 0 or passed < total:
            return False, passed, (total or total0), result
    return True, last_total, last_total, last


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


def _remaining_failures(verdict: Dict[str, Any]) -> int:
    """The count of GENUINE failing checks for a round's best attempt — what the next attempt must
    still fix. Zero iff fully verified. Quarantined (flawed) tests are already excluded from
    passed/total, so this counts only real failures: missing visible passes + missing held-out
    passes + a failed delivery/output gate. Drives stall detection (no reduction -> stalling)."""
    if verdict.get("verified"):
        return 0
    total = int(verdict.get("total", 0) or 0)
    passed = int(verdict.get("passed", 0) or 0)
    htotal = int(verdict.get("hidden_total", 0) or 0)
    hpassed = int(verdict.get("hidden_passed", 0) or 0)
    rem = max(0, total - passed) + max(0, htotal - hpassed)
    if verdict.get("gate_fail"):
        rem += 1
    return rem or 1          # not verified -> at least one outstanding failure


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


def _heldout_frac(att: Attempt) -> float:
    """Fraction of held-out/invariant checks this attempt passed (0.0 when none ran). Lets selection
    prefer the attempt that GENERALISES better when the visible-test score ties, so the final verdict
    reflects the latest, best-generalising attempt rather than a stale earlier one with the same visible
    score but a worse held-out tally."""
    v = att.verdict
    ht = int(v.get("hidden_total", 0))
    return (int(v.get("hidden_passed", 0)) / ht) if ht > 0 else 0.0


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
def run_agent(task: str = "", *, brief: str = "", max_iters: Optional[int] = None,
              use_search: bool = True, directive_path: Optional[str] = None,
              conversation: str = "", on_event: Optional[OnEvent] = None,
              result_memory: Optional[Any] = None, user_id: str = "local") -> AgentResult:
    emit: OnEvent = on_event or (lambda e: None)
    task = (task or "").strip()
    brief = (brief or "").strip()
    if not task and brief:
        task = "Achieve the goal described in the brief."
    if not task:
        return AgentResult(task, False, "", "", "No task given.", [])

    # Iterate until FULLY verified, the attempt cap is hit, or progress stalls. An explicit
    # max_iters (CLI --iters, tests) is an exact cap; otherwise use the configurable AGENT_MAX_ATTEMPTS.
    budget = int(max_iters) if max_iters is not None else max_attempts()
    budget = max(1, budget)
    stall_cap = stall_limit()

    # RESULT-MEMORY REUSE: seed the first attempt with a near-identical VERIFIED prior solution so the
    # agent ADAPTS proven code (faster, higher first-attempt success). It is still re-verified through
    # the FULL gate stack below — reuse never bypasses a gate, and only verified runs are ever reused.
    seed_code = ""
    if result_memory is not None and result_memory_enabled():
        try:
            prior = result_memory.find_verified_solution(user_id=user_id, task=task)
        except Exception:
            prior = None
        if prior and (prior.get("code") or "").strip():
            seed_code = prior["code"]
            try:
                result_memory.record_agent_run_reuse(int(prior["id"]))
            except Exception:
                pass
            emit({"type": "reuse", "message":
                  "Adapting a previously verified solution for a near-identical task "
                  f"({int(prior.get('similarity', 0.0) * 100)}% match) — it will be re-verified."})

    # Resilient model chain: the user's selection first (AGENT_MODEL or the chat's OPENAI_MODEL),
    # then configured fallbacks. On a 429/timeout/5xx the provider retries + switches to another
    # AVAILABLE model rather than failing the request — never overriding the user's choice.
    model_chain = _agent_model_chain()
    provider = ResilientProvider(model_chain, emit)
    if not provider.is_available:
        emit({"type": "error", "message": provider.unavailable_message()})
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
    agent_trace = tracing.start_trace("agent_run", max_iters=budget, use_search=bool(use_search))

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

    # Test-VALIDATION oracle: for non-simulation this IS the expected-value oracle above. For a
    # SIMULATION there is no exact-value oracle (chaotic trajectories diverge), but an INDEPENDENT
    # correct reference STILL validates the property/invariant tests — a check the correct reference
    # also fails is flawed (a too-tight tolerance or a wrong definition, e.g. asserting total momentum
    # ~= 0 when it is conserved at a nonzero value), so it is quarantined; a check it passes that the
    # candidate fails is a REAL bug. Built for VALIDATION ONLY — the tests stay property-based
    # (use_reference is False for simulation), so no expected value is ever derived from this oracle.
    validation_oracle = oracle
    if (not validation_oracle and task_type == "simulation"
            and test_validation_enabled() and reference_tests_enabled()):
        emit({"type": "status", "message": "Building an independent reference to validate the checks…"})
        with agent_trace.span("validation_reference") as _sp:
            validation_oracle = _generate_reference(provider, task, requirements, task_type)
            _sp.set(chars=len(validation_oracle or ""))
        emit({"type": "reference", "scope": "validation",
              "chars": len(validation_oracle or ""), "used": bool((validation_oracle or "").strip())})

    # (b) Generate concrete correctness tests for THIS task — the acceptance criteria. With the
    # oracle available, the tests compute expected via ref.* at runtime instead of guessing.
    emit({"type": "status", "message": "Writing correctness tests…"})
    with agent_trace.span("tests") as _sp:
        tests = _generate_tests(provider, task, requirements, task_type=task_type,
                                use_reference=use_reference)
        _sp.set(count=_count_tests(tests))

    # (b.1) TEST-CRITIC: a SEPARATE role audits the generated suite and REWRITES any invalid test into a
    # check of the TRUE requirement BEFORE it can judge a candidate. Runs ONCE on the visible suite; the
    # repaired suite is then still validated against the reference below (defence in depth). Fail-open.
    if test_critic_enabled():
        emit({"type": "status", "message": "Test-critic auditing the correctness tests…"})
        _crit = _critique_tests(provider, task, requirements, tests, validation_oracle,
                                "correctness tests")
        if _crit != tests:
            emit({"type": "test_critic", "scope": "visible",
                  "before": _count_tests(tests), "after": _count_tests(_crit)})
            tests = _crit

    test_n = _count_tests(tests)
    emit({"type": "tests", "iteration": 0, "code": tests, "count": test_n})

    # Oracle test-validation: a generated test the KNOWN-CORRECT oracle itself fails is invalid — its
    # expected value was guessed, its tolerance is too tight for the method, or it checks the wrong
    # quantity. Validate the visible suite against the oracle ONCE and QUARANTINE such tests; they are
    # excluded from every candidate's pass/total, so correct code is never falsely failed while every
    # oracle-passing test still gates genuinely wrong code.
    visible_quarantine: set = set()
    if test_validation_enabled() and validation_oracle:
        emit({"type": "status", "message": "Validating the tests against the reference oracle…"})
        visible_quarantine = _invalid_tests(validation_oracle, tests)
        if visible_quarantine:
            emit({"type": "test_validation", "scope": "visible",
                  "quarantined": sorted(visible_quarantine),
                  "message": (f"Quarantined {len(visible_quarantine)} visible test(s) the reference "
                              "itself fails (invalid expected/tolerance/definition): "
                              + ", ".join(sorted(visible_quarantine)))})

    # Completeness gate prep: extract the explicit deliverables the request asks for, ONCE (only when
    # the task asks for output and the gates are on). Checked against the real stdout each round.
    deliverables: List[str] = []
    gate_output = delivery_gates_enabled() and _wants_output(task)
    if gate_output:
        deliverables = _extract_deliverables(provider, task, requirements)
        emit({"type": "deliverables", "items": deliverables})

    attempts: List[Attempt] = []
    best: Optional[Attempt] = None
    best_clean: Optional[Attempt] = None      # best NON-cheating attempt — the only thing we return
    last_code = seed_code     # seed from a near-identical VERIFIED prior solution (re-verified below)
    feedback = ""
    frozen_clause = ""        # _freeze_clause: keep already-passing parts, fix only the failing ones
    frozen_best: set = set()  # the most checks ever seen green — used to detect a regression
    rounds_failed = 0
    cheat_count = 0
    best_remaining = None     # fewest GENUINE failing checks seen so far (for stall detection)
    stall = 0                 # consecutive rounds with no reduction in remaining failures
    stop_reason = "max_attempts"
    prev_failing: set = set()    # failing checks from the prior round — to spot a non-moving fix
    prev_diagnosis = ""          # the prior round's root-cause diagnosis — vary it if it didn't help
    mem = _AttemptMemory()
    hstate: Dict[str, Any] = {"code": None, "strict": False, "quarantine": set(),
                              "nonunique": set()}                                    # lazy held-out
    _heldout_lock = threading.Lock()                            # parallel candidates share it

    def _ensure_heldout(hp, strict: bool) -> str:
        """(C1/C3) Build (once, cached) the held-out suite = hidden tests + invariants + DEFINITION-
        MATCH checks (one per requested output, asserting the reported value matches the request's
        exact definition — independent of the candidate and its reference). Rebuilt stricter if
        escalation flips `strict`. Never shown to the solver. Thread-safe: parallel candidates that
        all pass the visible tests build it exactly once. The definition gate runs even when the
        hidden-tests gate is off, so 'right logic, wrong reported answer' is always caught."""
        if not (hidden_tests_enabled() or definition_gate_enabled()):
            return ""
        with _heldout_lock:
            if hstate["code"] is not None and hstate["strict"] == strict:
                return hstate["code"]
            emit({"type": "status", "message": "Building held-out hidden / definition checks…"})
            hidden = invariants = definitions = ""
            if hidden_tests_enabled():
                hidden = _generate_hidden_tests(hp, task, requirements, strict=strict,
                                                task_type=task_type, use_reference=use_reference)
                invariants = _generate_invariants(hp, task, requirements, strict=strict,
                                                  task_type=task_type)
            if definition_gate_enabled():
                definitions = _generate_definition_checks(hp, task, requirements, strict=strict,
                                                          task_type=task_type)
            combined = "\n\n".join(p for p in (hidden, invariants, definitions) if (p or "").strip())
            # TEST-CRITIC on the held-out suite (incl. DEFINITION checks the execution backstop exempts):
            # audit + repair each invalid check ONCE before it can judge a candidate. Fail-open.
            if test_critic_enabled() and (combined or "").strip():
                _crit = _critique_tests(hp, task, requirements, combined, validation_oracle,
                                        "held-out hidden / invariant / definition checks")
                if _crit != combined:
                    emit({"type": "test_critic", "scope": "heldout",
                          "before": _count_tests(combined), "after": _count_tests(_crit)})
                    combined = _crit
            # Oracle test-validation on the held-out suite: quarantine the hidden/invariant checks the
            # known-correct reference itself fails — a guessed expected value, a too-tight tolerance, or
            # a property THIS task does NOT have (an idealised invariant broken by a dissipative /
            # driving / open / injecting / asymmetric / random term: "requirement-not-implied-by-task").
            # An invalid check must never falsely fail a correct candidate; definition checks are exempt
            # (oracle-independent by design). Uses the VALIDATION oracle, so this also covers simulation
            # tasks (no exact-value oracle) — provided the reference faithfully models the real system.
            quarantine: set = _heldout_quarantine(validation_oracle, combined)
            if quarantine:
                emit({"type": "test_validation", "scope": "heldout",
                      "quarantined": sorted(quarantine),
                      "message": (f"Quarantined {len(quarantine)} held-out check(s) the reference itself "
                                  "fails (invalid expected value / a property this task does not have): "
                                  + ", ".join(sorted(quarantine)))})
            # Carry the non-unique-exact quarantine across a stricter held-out rebuild too.
            hstate["code"], hstate["strict"] = combined, strict
            hstate["quarantine"] = quarantine | hstate.get("nonunique", set())
            emit({"type": "heldout", "count": _count_tests(combined), "strict": strict})
            return combined

    # NON-UNIQUE quantity validation (extends the quarantine, before the loop). Many correct answers
    # are defined only up to scaling/sign/ordering/phase/basis; a test asserting exact equality to one
    # such representation passes the self-agreeing oracle but FAILS a different valid solution. Detect
    # those by executing an INDEPENDENT cross-reference (validated against the property checks) and
    # quarantine the exact tests it fails that the oracle passes. Gated + fail-open.
    if (nonunique_validation_enabled() and use_reference and (oracle or "").strip()
            and test_validation_enabled()):
        _heldout0 = _ensure_heldout(provider, False)        # property checks live here (cached for the loop)
        _all_tests = (tests + "\n\n" + _heldout0) if (_heldout0 or "").strip() else tests
        _nu = _nonunique_exact_tests(provider, task, requirements, task_type, oracle,
                                     _all_tests, verify_seeds())
        if _nu:
            visible_quarantine |= _nu
            with _heldout_lock:
                hstate["nonunique"] = set(hstate.get("nonunique") or set()) | _nu
                hstate["quarantine"] = set(hstate.get("quarantine") or set()) | _nu
            emit({"type": "test_validation", "scope": "nonunique", "quarantined": sorted(_nu),
                  "message": (f"Quarantined {len(_nu)} test(s) asserting exact equality on a NON-UNIQUE "
                              "quantity (defined up to scaling/sign/ordering/phase/basis) — a valid "
                              "alternative solution fails them: " + ", ".join(sorted(_nu)))})

    attempts_taken = 0
    for i in range(1, budget + 1):
        attempts_taken = i
        directive = _read_directive(directive_path)
        if directive:
            emit({"type": "directive", "iteration": i, "text": directive[:300]})
            feedback = (feedback + "\nUSER DIRECTIVE (priority): " + directive).strip()

        # (6) Escalate after two failed rounds OR two cheating catches; two cheats also strengthens
        # the held-out audit. Escalation RESPECTS the user's selected model — it only lets
        # AGENT_MODEL_STRONG lead when no model was explicitly selected; otherwise it keeps the
        # user's chain (the resilient provider still falls back on errors).
        strict = cheat_count >= 2
        gen_provider = provider
        if rounds_failed >= 2 or strict:
            esc_chain = _escalated_chain(model_chain)
            if esc_chain != model_chain:
                gen_provider = ResilientProvider(esc_chain, emit)
                emit({"type": "status",
                      "message": f"Escalating to a stronger model ({esc_chain[0]})…"})

        emit({"type": "think", "iteration": i,
              "message": f"Writing code to pass the tests (attempt {i}/{budget})…"})

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
                                      temperature=temperature, variant=variant, frozen=frozen_clause,
                                      wants_output=gate_output)
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
                if visible_quarantine:          # judge on valid (oracle-passing) tests only
                    vc = _valid_counts(result.stdout or "", visible_quarantine)
                    if vc is not None:
                        passed, total = vc

            relevant = _is_relevant_code(task, code, tests)        # (C6) algorithm-match gate
            verdict = _verdict_from_tests(passed, total, relevant, result)
            # Per-check pass/fail names so the next attempt can FREEZE what already passes.
            verdict["passing_checks"] = _passing_names(result.stdout or "")
            # failing_checks must AGREE with the quarantine-adjusted passed/total: a test the reference
            # oracle itself fails is invalid, so a candidate "failing" it is not a real failure — exclude
            # it so it is never reported, fed to the diagnosis, or logged as a failure pattern.
            verdict["failing_checks"] = _genuine_failing(result.stdout or "", visible_quarantine)
            cheating = bool(cheat and cheat.flagged)
            verdict["cheating"] = cheating
            verdict["cheat_reasons"] = list(cheat.reasons) if cheat else []
            verdict["verified"] = False

            if cheating:
                verdict["done"] = False
                verdict["feedback"] = (
                    "Your solution was REJECTED as reward-hacking: " + "; ".join(cheat.reasons)
                    + ". Do NOT hardcode outputs, special-case the example inputs, read the tests, or "
                    "fake the functions; and do NOT force a conserved quantity to pass by renormalising "
                    "the state every step or clamping the output — fix the SCHEME so the quantity "
                    "EMERGES from correct dynamics. Solve the GENERAL task.")
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
                            code, heldout, verify_seeds(), reference_src=oracle,
                            quarantine=hstate.get("quarantine"))
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
            with _rc.ContextThreadPoolExecutor(max_workers=n_cand) as _ex:   # workers inherit request settings
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

        if round_best is None:         # all candidates were empty or produced no code: a provider
            rounds_failed += 1         # hiccup (rate-limit, blank/unparseable reply), NOT genuine
            continue                   # non-progress -> retry within the budget, don't trip the stall

        attempts.append(round_best)
        if best is None or _score(round_best) > _score(best):
            best = round_best
        if not round_best.verdict.get("cheating"):
            # Strictly higher score wins; on a TIE prefer the attempt that passes MORE held-out checks
            # (generalises better) so the reported verdict reflects the latest/best attempt, not a stale
            # earlier one with the same visible score but a worse held-out tally.
            if (best_clean is None or _score(round_best) > _score(best_clean)
                    or (_score(round_best) == _score(best_clean)
                        and _heldout_frac(round_best) > _heldout_frac(best_clean))):
                best_clean = round_best

        # (1/4) DELIVERY gates: a candidate that passed the visible + held-out gates, on a task that
        # asks for output, must ACTUALLY RUN and PRINT every requested deliverable. Run its demo once,
        # capture real stdout, and downgrade it (-> regenerate next round) if it produced nothing
        # (execution gate) or dropped a requested output (completeness gate).
        if gate_output and round_best.verdict.get("verified"):
            emit({"type": "status", "message": "Running the solution to check its real output…"})
            out, missing = _capture_and_check(round_best.code, deliverables)
            round_best.verdict["demo_output"] = out
            _apply_output_gates(round_best.verdict, wants_output=True, output=out, missing=missing)
            if round_best.verdict.get("gate_fail"):
                emit({"type": "gate_fail", "iteration": i,
                      "reason": round_best.verdict["gate_fail"]})

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
        elif v.get("gate_fail"):
            mem.add(f"iter {i}: passed tests but FAILED a delivery gate — {(v.get('gate_fail') or '')[:160]}")
        elif not v.get("verified"):
            mem.add(f"iter {i}: only {v.get('passed')}/{v.get('total')} visible tests passed")

        last_code = round_best.code
        if v.get("verified"):
            stop_reason = "verified"
            break
        if v.get("relevant") is False:
            emit({"type": "status",
                  "message": "Off-topic for the requested algorithm — regenerating…"})
        feedback = v.get("feedback", "")
        rounds_failed += 1

        # ROOT-CAUSE DIAGNOSIS: before the next rewrite, map the symptom (the failing checks + the
        # traceback) to the likely MECHANISM and code location, so the next attempt targets the cause
        # instead of mutating blindly. If the same checks keep failing, the diagnosis is told to pick
        # a DIFFERENT mechanism. Prepended to `feedback`; fail-open (a hiccup keeps the raw feedback).
        cur_failing = set(v.get("failing_checks") or [])
        diagnosis = _diagnose_failure(
            gen_provider, task, requirements, round_best.code, feedback, sorted(cur_failing),
            prev_diagnosis, repeated=bool(prev_failing and (cur_failing & prev_failing)))
        if diagnosis:
            round_best.verdict["diagnosis"] = diagnosis
            emit({"type": "diagnosis", "iteration": i, "message": diagnosis[:400]})
            feedback = ("DIAGNOSIS AGENT — root-cause analysis + FIX DIRECTIVE. APPLY this mechanism "
                        "fix at its source (do NOT mask or merely loosen tolerances); the gates, not "
                        "you, decide when it is done:\n" + diagnosis + "\n\n" + feedback)
            prev_diagnosis = diagnosis
        prev_failing = cur_failing

        # FREEZE for the NEXT attempt: keep the code that satisfies the already-passing checks and
        # revise only the failing ones. Flag any check that was green before but this round re-broke
        # (a regression) so the next attempt restores it instead of trading one pass for another.
        round_passing = set(v.get("passing_checks") or [])
        regressed = sorted(frozen_best - round_passing)
        if regressed:
            mem.add(f"iter {i}: RE-BROKE previously-passing {', '.join(regressed[:6])} — restore them")
        frozen_best |= round_passing
        frozen_clause = _freeze_clause(sorted(round_passing), regressed)

        # Stall detection: keep iterating only while attempts REDUCE the genuine failing checks.
        # No reduction for AGENT_STALL_LIMIT consecutive rounds -> stop with the best effort so far,
        # rather than burning the whole attempt budget when more tries clearly are not helping.
        rem = _remaining_failures(v)
        if best_remaining is None or rem < best_remaining:
            best_remaining, stall = rem, 0
        else:
            stall += 1
        if stall >= stall_cap:
            stop_reason = "stall"
            emit({"type": "status", "message":
                  f"No progress for {stall} attempt(s) ({rem} genuine check(s) still failing) — "
                  "stopping with the best result so far."})
            break

    # (7) Honest outcome — prefer the best NON-cheating attempt; never present a gaming solution.
    try:
        from backend.agent.reference_code import topic_of
        topic = topic_of(task) or task
    except Exception:
        topic = task

    final = best_clean if best_clean is not None else best

    # RE-VALIDATE AGAINST THE LATEST ATTEMPT (staleness fix): don't settle for a STALE 'partial'. The
    # chosen best_clean can be an EARLIER attempt demoted for 'missing output', while the MOST RECENT
    # attempt actually prints every requested value. Re-run the latest delivery-eligible attempt ONCE
    # on fresh stdout; if it is now complete, present IT as verified (the earlier failure was judged
    # against a stale state). Re-validates by RE-RUNNING real code — never resurrects a held-out /
    # visible / cheating / off-topic failure (those are excluded from the upgrade).
    if best_clean is not None and not final.verdict.get("verified"):
        fresh = _latest_revalidated_delivery(attempts, deliverables, gate_output)
        if fresh is not None:
            emit({"type": "status", "message": "Re-validated the latest attempt — it produces every "
                  "requested output; clearing the stale 'missing output' and marking it verified."})
            best_clean = final = fresh

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

    tries = f" after {attempts_taken} attempt{'s' if attempts_taken != 1 else ''}"
    if final is None:
        verification, answer, present = "failed", "The agent could not produce a working solution.", None
    elif best_clean is None:                  # every attempt was flagged for gaming
        verification = "rejected_cheating"
        answer = ("Rejected — possible test gaming was detected in every attempt, so no genuine, "
                  "verified solution could be produced.")
        present = None                        # never present the gaming code as a deliverable
    elif final.verdict.get("verified"):
        verification = "verified"
        gates = (f" plus {htotal} held-out hidden/spec/invariant checks on {verify_seeds()} random seeds"
                 if htotal else "")
        outclause = (" and produced every requested output when run"
                     if (gate_output and (final.verdict.get("demo_output") or "").strip()) else "")
        answer = (f"Implemented {topic} in Python — passes all {btotal} visible tests{gates}"
                  f"{outclause} (fully verified{tries}).")
        present = final
    else:
        # Honest partial: never a fake "verified". Say how many GENUINE checks pass, why the rest
        # remain, how many attempts ran, and whether we stopped early (stall) or hit the cap.
        verification = "partial"
        why = (final.verdict.get("diagnosis") or final.verdict.get("gate_fail")
               or (final.verdict.get("feedback") or "").split("\n")[0])
        why = (" — " + why.strip()[:160]) if (why or "").strip() else ""
        stop = (" — stopped early, no further progress" if stop_reason == "stall"
                else f" — reached the {budget}-attempt limit" if stop_reason == "max_attempts" else "")
        answer = (f"Best effort at {topic} in Python — {bpassed}/{btotal} genuine checks pass "
                  f"(partially verified{tries}{stop}){why}.")
        present = final

    # (5) Execution output: if the request asks to print/show/return a result, RUN the finished
    # solution with a small driver and capture its REAL stdout — the actual values, not test noise.
    best_output = ""
    if present is not None and _wants_output(task):
        cached = (present.verdict.get("demo_output") or "").strip()    # captured by the delivery gate
        if cached:
            best_output = clip_keep_ends(cached, DEMO_OUTPUT_CAP)
            emit({"type": "output", "text": best_output})
        else:
            try:
                emit({"type": "status", "message": "Running the solution to capture its output…"})
                driver = _generate_demo_driver(provider, task, requirements, present.code)
                if (driver or "").strip():
                    dres = run_python_auto(present.code + "\n\n# === demo run ===\n" + driver)
                    if dres.ok and (dres.stdout or "").strip():
                        best_output = clip_keep_ends((dres.stdout or "").strip(), DEMO_OUTPUT_CAP)
                        emit({"type": "output", "text": best_output})
            except Exception:                       # noqa: BLE001 - output is a bonus; never break
                best_output = ""

    res = AgentResult(
        task=task,
        success=(verification == "verified"),
        best_code=present.code if present else "",
        best_output=best_output,
        answer=answer,
        attempts=attempts,
        tests_passed=bpassed,
        tests_total=btotal,
        verification=verification,
        hidden_passed=hpassed,
        hidden_total=htotal,
        cheat_flags=cheat_flags,
        attempts_taken=attempts_taken,
        stop_reason=(stop_reason if final is not None else "failed"),
    )
    emit({"type": "final", "success": res.success, "verification": verification,
          "answer": res.answer, "code": res.best_code, "output": res.best_output,
          "iterations": len(attempts), "attempts_taken": attempts_taken,
          "stop_reason": res.stop_reason, "tests_passed": bpassed, "tests_total": btotal,
          "hidden_passed": hpassed, "hidden_total": htotal})
    agent_trace.set(success=res.success, iterations=len(attempts), verification=verification,
                    tests_passed=bpassed, tests_total=btotal).end()

    # RESULT MEMORY: record this run's outcome (verified / partial / failed) for the failure-pattern
    # report and for verified-only reuse. Best-effort — a store hiccup never breaks the run.
    if result_memory is not None and result_memory_enabled():
        fv = final.verdict if final is not None else {}
        try:
            result_memory.record_agent_run(
                user_id=user_id, task=task, code=res.best_code, output=res.best_output,
                verification=verification, requirements=requirements, task_type=task_type,
                tests_passed=bpassed, tests_total=btotal, hidden_passed=hpassed, hidden_total=htotal,
                attempts_taken=attempts_taken, stop_reason=res.stop_reason,
                cheat_reasons=list(fv.get("cheat_reasons") or []),
                diagnosis=fv.get("diagnosis") or "", gate_fail=fv.get("gate_fail") or "",
                failing_checks=list(fv.get("failing_checks") or []))
        except Exception:
            pass
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
    attempts_taken = int(getattr(res, "attempts_taken", 0) or 0)
    tries = f" after {attempts_taken} attempt{'s' if attempts_taken != 1 else ''}" if attempts_taken else ""
    # A gaming solution is NEVER presented as a clean answer.
    if verification == "rejected_cheating":
        return ("> ⛔ Rejected — possible test gaming detected; no genuine, verified solution was "
                "produced. Try rephrasing the request, or set a stronger model (AGENT_MODEL_STRONG).")
    if verification == "partial" or (total and not (getattr(res, "success", False) and passed >= total)):
        parts.append(f"> ⚠ Partially verified — {passed}/{total} genuine checks passing{tries}.")
    if answer:
        parts.append(answer)
    if code:
        parts.append(f"```python\n{code}\n```")
    if output:
        parts.append(f"**Output:**\n```text\n{output}\n```")
    return "\n\n".join(parts) or "_(the agent produced no result)_"
