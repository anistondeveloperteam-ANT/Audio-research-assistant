"""
The research agent's THINK -> EXECUTE -> REFLECT loop.

    THINK    : the LLM designs a complete Python program for the task.
    EXECUTE  : the program is run in a throwaway Docker sandbox (code_runner).
    REFLECT  : the LLM reviews the *actual run result* and decides: done, or refine.

It keeps the best working attempt and stops when the reviewer is satisfied (or after
`max_iters`). Optionally it first searches the web/papers/GitHub for relevant
approaches to inform the first attempt.

Two ideas adapted (original code) from `auto-deep-researcher-24x7` (Apache-2.0):
  - the THINK -> EXECUTE -> REFLECT control loop, and
  - a constant-size two-tier memory (`memory.py`) so many cycles never bloat context,
    steered by a PROJECT_BRIEF (the goal + an if-then decision tree) plus an optional
    mid-flight HUMAN_DIRECTIVE file.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.agent.code_runner import RunResult, docker_available, run_python
from backend.agent.hooks import pre_run
from backend.agent.memory import TwoTierMemory
from backend.llm.streaming_provider import get_provider

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
    "You are an expert algorithms engineer. Given a task, write ONE complete, "
    "self-contained Python program that solves it and PRINTS the final answer/result "
    "clearly (plus any benchmark numbers or a short correctness check). "
    "Rules: standard library only unless you are certain a package is installed; if an "
    "import would fail, use the stdlib instead. No network, no reading files, no input(). "
    "The program must run to completion within a few seconds and print its result. "
    "Output ONLY the Python code — no explanation, no markdown."
)

_REFLECT_SYSTEM = (
    "You are a strict reviewer. You are given a task, a candidate Python program, and "
    "the ACTUAL result of running it in a sandbox. Judge whether it correctly and well "
    "solves the task. Reply with ONLY a JSON object and nothing else:\n"
    '{"success": true|false, "score": 0-100, "done": true|false, '
    '"feedback": "concrete fix or improvement if not done", '
    '"answer": "the final answer/best algorithm in 1-2 sentences"}\n'
    "success = it ran and is correct. done = true only if it is correct AND good enough "
    "that further iteration would not meaningfully improve it."
)


def _generate_code(provider, memory_context: str, last_code: str, directive: str) -> str:
    """THINK: design (or refine) the program from the constant-size memory context.

    `memory_context` = brief (goal/decision-tree) + a compacted log of prior attempts.
    Only the single most-recent full program is carried verbatim (for line-level
    refinement) so the prompt stays bounded as iterations grow.
    """
    parts = [memory_context]
    if last_code:
        parts.append(f"\nYOUR PREVIOUS PROGRAM (improve it):\n```python\n{last_code}\n```")
    if directive:
        parts.append(f"\nNEW INSTRUCTION FROM THE USER (take priority):\n{directive}")
    parts.append("\nWrite the complete, improved program now.")
    return _extract_code(_complete(provider, _GEN_SYSTEM, "\n".join(parts), GEN_MAX_TOKENS))


def _reflect(provider, task: str, code: str, result: RunResult) -> Dict[str, Any]:
    user = (
        f"TASK:\n{task}\n\n"
        f"PROGRAM:\n```python\n{code}\n```\n\n"
        f"RUN RESULT: {result.summary}\n"
        f"STDOUT:\n{result.stdout or '(none)'}\n\n"
        f"STDERR:\n{result.stderr or '(none)'}\n"
        f"{('HARNESS ERROR: ' + result.error) if result.error else ''}"
    )
    verdict = _parse_json(_complete(provider, _REFLECT_SYSTEM, user, REFLECT_MAX_TOKENS))
    # Defensive defaults if the model didn't return clean JSON.
    verdict.setdefault("success", bool(result.ok))
    verdict.setdefault("score", 55 if result.ok else 0)
    verdict.setdefault("done", False)
    verdict.setdefault("feedback", "" if result.ok else (result.stderr or result.error or "It failed to run."))
    verdict.setdefault("answer", "")
    return verdict


def _score(att: Attempt) -> int:
    # A program that actually ran always beats one that didn't; then rank by review score.
    base = 1000 if att.result.ok else 0
    try:
        return base + int(att.verdict.get("score", 0))
    except Exception:
        return base


def _gather_context(task: str, on_event: OnEvent) -> str:
    """Best-effort: search the web/papers/GitHub for relevant approaches. Never fatal."""
    try:
        from backend.external_search.orchestrator import gather_external_evidence
        sources, _ = gather_external_evidence(task, max_results=6)
    except Exception as exc:
        on_event({"type": "warning", "message": f"Research step skipped: {exc}"})
        return ""
    lines: List[str] = []
    for s in sources[:6]:
        snippet = (getattr(s, "text", "") or getattr(s, "snippet", "") or "").strip().replace("\n", " ")
        if snippet:
            lines.append(f"- {getattr(s, 'title', 'source')}: {snippet[:300]}")
    return "\n".join(lines)


def _build_brief(task: str, brief: str, context: str) -> str:
    """Tier-1 brief: the user's PROJECT_BRIEF if given, else a goal built from the task,
    plus any research context. TwoTierMemory clips this to its cap."""
    head = brief.strip() if brief.strip() else f"# Goal\n{task}"
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
              on_event: Optional[OnEvent] = None) -> AgentResult:
    emit: OnEvent = on_event or (lambda e: None)
    task = (task or "").strip()
    brief = (brief or "").strip()
    if not task and brief:
        task = "Achieve the goal described in the brief."
    if not task:
        return AgentResult(task, False, "", "", "No task given.", [])

    provider = get_provider()
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

    context = ""
    if use_search:
        emit({"type": "status", "message": "Researching relevant approaches…"})
        context = _gather_context(task, emit)
        emit({"type": "context", "chars": len(context)})

    # Tier-1 brief (frozen) + Tier-2 log (auto-compacting) -> constant-size context.
    memory = TwoTierMemory(brief=_build_brief(task, brief, context))
    attempts: List[Attempt] = []
    best: Optional[Attempt] = None
    last_code = ""

    for i in range(1, max_iters + 1):
        directive = _read_directive(directive_path)
        if directive:
            emit({"type": "directive", "iteration": i, "text": directive[:300]})
        emit({"type": "think", "iteration": i, "message": f"Designing a solution (attempt {i}/{max_iters})…"})
        code = _generate_code(provider, memory.context(), last_code, directive)
        emit({"type": "code", "iteration": i, "code": code})

        # Pre-execution lifecycle hook (kimi-code idea): audit + allow/block gate.
        gate = pre_run(code, task=task)
        if not gate.allowed:
            emit({"type": "blocked", "iteration": i, "reason": gate.reason})
            result = RunResult(False, -1, "", "", 0.0, f"blocked by policy: {gate.reason}")
        else:
            emit({"type": "run", "iteration": i, "message": "Running it in the Docker sandbox…"})
            result = run_python(code)
        emit({"type": "run_result", "iteration": i, "ok": result.ok, "summary": result.summary,
              "stdout": result.stdout, "stderr": result.stderr, "error": result.error})

        verdict = _reflect(provider, task, code, result)
        emit({"type": "reflect", "iteration": i, "verdict": verdict})

        att = Attempt(i, code, result, verdict)
        attempts.append(att)
        if best is None or _score(att) > _score(best):
            best = att

        # Record a compact, bounded note of this attempt for future cycles.
        memory.append(f"Attempt {i}: {result.summary}. Review score={verdict.get('score')} "
                      f"done={verdict.get('done')}. {(verdict.get('feedback') or '')[:240]}")
        last_code = code
        if result.ok and verdict.get("done"):
            break

    res = AgentResult(
        task=task,
        success=bool(best and best.result.ok and best.verdict.get("success")),
        best_code=best.code if best else "",
        best_output=best.result.stdout if best else "",
        answer=(best.verdict.get("answer", "") if best else ""),
        attempts=attempts,
    )
    emit({"type": "final", "success": res.success, "answer": res.answer,
          "code": res.best_code, "output": res.best_output, "iterations": len(attempts)})
    return res
