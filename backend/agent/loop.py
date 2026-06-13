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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


# ----------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------
def _complete(provider, system: str, user: str, max_tokens: int) -> str:
    return "".join(provider.stream_chat(
        [{"role": "user", "content": user}], system=system,
        max_tokens=max_tokens, temperature=0.2,
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

_TESTS_SYSTEM = (
    "You write rigorous but ROBUST unit tests as plain Python (no pytest, no unittest). Given a "
    "task and its requirements, write 5-7 focused test functions named test_<name>() that:\n"
    "- call the SOLUTION's functions directly (they are defined in the same file);\n"
    "- use SELF-CONSISTENT inputs: the SAME calling convention and array shapes in EVERY test, "
    "matching ONE function signature — do NOT require the function to accept several different "
    "input shapes;\n"
    "- compare floats with tolerances (math.isclose, or numpy.allclose with explicit rtol/atol), "
    "NEVER exact ==; build small CONCRETE inputs (use numpy if it helps);\n"
    "- prefer PROPERTIES that hold for ANY correct implementation over hard-coded magic numbers "
    "(e.g. for an MVDR beamformer: the distortionless constraint w^H d ~= 1, and output noise "
    "power <= input noise power; for Black-Scholes: put-call parity C - P ~= S - K*exp(-rT), "
    "monotonicity in volatility, and ONE textbook reference value);\n"
    "- each test must `assert` and raise AssertionError on failure.\n"
    "Do NOT define the solution or any test runner. Output ONLY the Python test functions."
)

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


def _generate_tests(provider, task: str, requirements: str) -> str:
    """(b) Generate 5-8 concrete test_* functions that target THIS task (derived, not hardcoded)."""
    user = (f"TASK:\n{task}\n\nREQUIREMENTS:\n{requirements}\n\n"
            "Write the test_* functions now (they call the solution's functions directly).")
    return _extract_code(_complete(provider, _TESTS_SYSTEM, user, GEN_MAX_TOKENS))


def _generate_solution(provider, task: str, requirements: str, tests: str, reference: str,
                       last_code: str, feedback: str) -> str:
    """(c) Write modular solution code so the provided tests pass. The tests are appended by the
    runner, not by the model."""
    parts = [f"TASK:\n{task}", f"\nREQUIREMENTS:\n{requirements}",
             "\nYour solution MUST define the functions these tests call so they pass. Do NOT "
             f"include the tests or a runner in your output:\n```python\n{tests}\n```"]
    if reference:
        parts.append(f"\nREFERENCE implementations (adapt the approach, do NOT copy):\n{reference[:3000]}")
    if last_code:
        parts.append(f"\nYOUR PREVIOUS SOLUTION (fix it):\n```python\n{last_code}\n```")
    if feedback:
        parts.append("\nThe tests FAILED last time. Read these PASS/FAIL lines and tracebacks and "
                     f"fix the SPECIFIC failures (do not rewrite from scratch):\n{feedback[:3000]}")
    parts.append("\nWrite the solution code now (functions only).")
    return _extract_code(_complete(provider, _GEN_SYSTEM, "\n".join(parts), GEN_MAX_TOKENS))


def _count_tests(tests_code: str) -> int:
    return len(re.findall(r"^\s*def\s+test_\w+\s*\(", tests_code or "", re.M))


def _run_against_tests(solution_code: str, tests_code: str):
    """Combine solution + generated tests + the runner, execute in the sandbox, and return
    (RunResult, passed, total). A crash before the runner -> 0 passed (stderr feeds the rewrite)."""
    script = solution_code + "\n\n# === generated tests ===\n" + tests_code + _TEST_FOOTER
    result = run_python_auto(script)
    m = re.search(r"TESTS_PASSED\s+(\d+)\s*/\s*(\d+)", result.stdout or "")
    passed = int(m.group(1)) if m else 0
    total = int(m.group(2)) if m else _count_tests(tests_code)
    return result, passed, total


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
    # Off-topic attempts never win; then a program that ran beats one that didn't; then score.
    if att.verdict.get("relevant") is False:
        return -1
    base = 1000 if att.result.ok else 0
    try:
        return base + int(att.verdict.get("score", 0))
    except Exception:
        return base


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
    if not docker_available():
        emit({"type": "error", "message": "Docker is not running — start Docker Desktop so the "
                                          "agent can run and verify its code."})
        return AgentResult(task, False, "", "", "Docker unavailable.", [])

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

    # (b) Generate concrete correctness tests for THIS task — the acceptance criteria.
    emit({"type": "status", "message": "Writing correctness tests…"})
    with agent_trace.span("tests") as _sp:
        tests = _generate_tests(provider, task, requirements)
        _sp.set(count=_count_tests(tests))
    test_n = _count_tests(tests)
    emit({"type": "tests", "iteration": 0, "code": tests, "count": test_n})

    strong_model = os.getenv("AGENT_MODEL_STRONG") or ""
    attempts: List[Attempt] = []
    best: Optional[Attempt] = None
    last_code = ""
    feedback = ""
    rounds_failed = 0

    for i in range(1, max_iters + 1):
        directive = _read_directive(directive_path)
        if directive:
            emit({"type": "directive", "iteration": i, "text": directive[:300]})
            feedback = (feedback + "\nUSER DIRECTIVE (priority): " + directive).strip()

        # (e) Escalate to a stronger model after two failed rounds, if one is configured.
        gen_provider = provider
        if rounds_failed >= 2 and strong_model:
            try:
                gen_provider = get_provider(strong_model)
                emit({"type": "status",
                      "message": f"Escalating to a stronger model ({strong_model})…"})
            except Exception:
                gen_provider = provider

        emit({"type": "think", "iteration": i,
              "message": f"Writing code to pass the tests (attempt {i}/{max_iters})…"})

        # (d) Best-of-2: up to two candidates per round; keep the higher-passing, on-topic one.
        round_best: Optional[Attempt] = None
        round_best_rank: Optional[tuple] = None
        for c in range(2):
            cand_feedback = feedback
            if c > 0:
                cand_feedback = (feedback + "\n\nYour first attempt this round did not pass all "
                                 "tests — take a materially different approach.").strip()
            with agent_trace.span("generate", iteration=i, candidate=c + 1) as _sp:
                code = _generate_solution(gen_provider, task, requirements, tests, reference,
                                          last_code, cand_feedback)
                _sp.set(code_len=len(code or ""))
            if not (code or "").strip():
                continue
            emit({"type": "code", "iteration": i, "candidate": c + 1, "code": code})

            # Pre-execution lifecycle hook (kimi-code idea): audit + allow/block gate. NEVER weakened.
            with agent_trace.span("prerun_hook", iteration=i, candidate=c + 1) as _sp:
                gate = pre_run(code, task=task)
                _sp.set(allowed=bool(gate.allowed), reason=gate.reason)
            if not gate.allowed:
                emit({"type": "blocked", "iteration": i, "candidate": c + 1, "reason": gate.reason})
                result = RunResult(False, -1, "", "", 0.0, f"blocked by policy: {gate.reason}")
                passed, total = 0, test_n
            else:
                emit({"type": "run", "iteration": i, "candidate": c + 1,
                      "message": "Running it against the tests in the Docker sandbox…"})
                with agent_trace.span("docker_run", iteration=i, candidate=c + 1) as _sp:
                    result, passed, total = _run_against_tests(code, tests)
                    _sp.set(ok=bool(result.ok), passed=passed, total=total,
                            seconds=round(result.seconds, 2))

            relevant = _is_relevant_code(task, code, tests)        # (C6) algorithm-match gate
            verdict = _verdict_from_tests(passed, total, relevant, result)
            att = Attempt(i, code, result, verdict)
            emit({"type": "run_result", "iteration": i, "candidate": c + 1, "ok": result.ok,
                  "passed": passed, "total": total, "relevant": relevant,
                  "summary": result.summary, "stdout": result.stdout, "stderr": result.stderr})

            rank = (1 if relevant else 0, passed, 1 if result.ok else 0)
            if round_best is None or rank > round_best_rank:
                round_best, round_best_rank = att, rank
            if verdict.get("done"):    # a relevant, fully-passing candidate ends the round early
                break

        if round_best is None:         # both candidates were empty or produced no code
            rounds_failed += 1
            continue

        attempts.append(round_best)
        if best is None or _score(round_best) > _score(best):
            best = round_best
        v = round_best.verdict
        emit({"type": "reflect", "iteration": i, "verdict": {
            "done": bool(v.get("done")), "relevant": bool(v.get("relevant")),
            "score": v.get("score"), "passed": v.get("passed"), "total": v.get("total"),
            "feedback": (v.get("feedback") or "")[:300]}})
        last_code = round_best.code
        if v.get("done"):
            break
        if v.get("relevant") is False:
            emit({"type": "status",
                  "message": "Off-topic for the requested algorithm — regenerating…"})
        feedback = v.get("feedback", "")
        rounds_failed += 1

    # (f) Honest outcome — derive a short answer from the best attempt's test tally.
    bpassed = int(best.verdict.get("passed", 0)) if best else 0
    btotal = int(best.verdict.get("total", 0)) if best else 0
    try:
        from backend.agent.reference_code import topic_of
        topic = topic_of(task) or task
    except Exception:
        topic = task

    # Optional automatic peer review of the best result — a relevance double-check only.
    try:
        from backend.answering.agentic_answer import auto_review_enabled
        from backend.answering.reviewer import review as _peer_review, is_relevant
        if best and auto_review_enabled():
            emit({"type": "status", "message": "Reviewing the best result…"})
            payload = f"Implements {topic}.\n\n```python\n{best.code or ''}\n```"
            intent = (convo + "\n\n" + task).strip() if convo else task
            rev = _peer_review(payload, task=intent)
            if rev and not rev.get("error") and not is_relevant(rev):
                best.verdict["relevant"] = False        # off-topic -> never "verified"
                best.verdict["done"] = False
                emit({"type": "reflect", "iteration": len(attempts), "verdict": {
                    "done": False, "relevant": False,
                    "feedback": "Peer review: off-topic for the request."}})
    except Exception:
        pass

    done = bool(best and best.verdict.get("done"))
    if not best:
        answer = "The agent could not produce a working solution."
    elif done:
        answer = (f"Implemented {topic} in Python — all {btotal} generated correctness "
                  "tests pass in the sandbox.")
    else:
        answer = (f"Best effort at {topic} in Python — {bpassed}/{btotal} generated correctness "
                  "tests pass (partially verified).")

    res = AgentResult(
        task=task,
        success=done,
        best_code=best.code if best else "",
        best_output=best.result.stdout if best else "",
        answer=answer,
        attempts=attempts,
        tests_passed=bpassed,
        tests_total=btotal,
    )
    emit({"type": "final", "success": res.success, "answer": res.answer,
          "code": res.best_code, "output": res.best_output, "iterations": len(attempts),
          "tests_passed": bpassed, "tests_total": btotal})
    agent_trace.set(success=res.success, iterations=len(attempts),
                    tests_passed=bpassed, tests_total=btotal).end()
    return res


def result_to_markdown(res) -> str:
    """Render an AgentResult as the markdown saved/shown for a coding turn: answer + code +
    output, with an honest verification label when tests didn't all pass. Shared by the chat
    code-route and the /api/agent persistence so both render identically."""
    parts = []
    answer = (getattr(res, "answer", "") or "").strip()
    code = (getattr(res, "best_code", "") or "").strip()
    output = (getattr(res, "best_output", "") or "").strip()
    total = int(getattr(res, "tests_total", 0) or 0)
    passed = int(getattr(res, "tests_passed", 0) or 0)
    if total and not (getattr(res, "success", False) and passed >= total):
        parts.append(f"> ⚠ Partially verified — {passed}/{total} generated tests passing.")
    if answer:
        parts.append(answer)
    if code:
        parts.append(f"```python\n{code}\n```")
    if output:
        parts.append(f"**Output:**\n```text\n{output}\n```")
    return "\n\n".join(parts) or "_(the agent produced no result)_"
