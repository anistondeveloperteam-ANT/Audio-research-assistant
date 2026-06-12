"""
Multi-agent code pipeline as an explicit LangGraph StateGraph.

The reasoning *steps* are the proven functions from loop.py / hooks.py / code_runner.py
/ reviewer.py / agentic_answer.py / retrieval / external_search — REUSED, not rewritten.
LangGraph only provides the structure: typed shared state, named nodes, concurrent
fan-out/fan-in, and a conditional edge that loops back under the configured budget.

    START ─▶ planner ─▶ ┌ fetcher_local  ┐ ─▶ coder ─▶ sandbox_runner ─▶ verifier ─▶ grader ─▶ END
                        └ fetcher_external┘                                              │
                          (concurrent)                (relevance‖citation‖code-run)      └▶ planner
                                                                                    (score<MIN & round<MAX)

Optional + fallback-safe: if `langgraph` is unavailable, `run_agent_graph` falls back to
the hand-rolled `run_agent` loop. Off by default (AGENT_GRAPH_ENABLED) — the existing
/api/agent path is unchanged unless the flag (or the queue) is turned on.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import operator
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from backend.agent.code_runner import RunResult, docker_available, run_python
from backend.agent.hooks import pre_run
from backend.agent.loop import (
    AgentResult, _build_brief, _generate_code, run_agent,
)
from backend.answering.agentic_answer import (
    max_verify_rounds, min_verify_score, verification_passed, verify_answer,
)
from backend.answering.reviewer import is_relevant, review
from backend.llm.streaming_provider import get_provider

logger = logging.getLogger(__name__)
OnEvent = Optional[Callable[[Dict[str, Any]], None]]

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB = ROOT / "data" / "agent_checkpoints.db"


def graph_available() -> bool:
    """True if the optional `langgraph` package is importable."""
    return importlib.util.find_spec("langgraph") is not None


# ----------------------------------------------------------------------
# Budgets (env-driven, enforced inside the graph; every violation is logged)
# ----------------------------------------------------------------------
class _Budget:
    def __init__(self) -> None:
        self.max_rounds = max_verify_rounds()                      # exact, shared with chat
        self.min_score = min_verify_score()                        # exact threshold
        try:
            self.max_tokens = int(os.getenv("MAX_TOKENS_PER_TASK", "4096"))
        except ValueError:
            self.max_tokens = 4096
        try:
            self.node_timeout = float(os.getenv("AGENT_NODE_TIMEOUT", "60"))
        except ValueError:
            self.node_timeout = 60.0
        self.tokens_used = 0

    def charge(self, *texts: str) -> None:
        """Best-effort token accounting (provider gives no exact usage): ~4 chars/token."""
        self.tokens_used += sum(len(t or "") for t in texts) // 4

    def tokens_exhausted(self) -> bool:
        over = self.tokens_used >= self.max_tokens
        if over:
            logger.warning("budget exceeded: tokens_used=%d >= MAX_TOKENS_PER_TASK=%d",
                           self.tokens_used, self.max_tokens)
        return over


def _make_checkpointer():
    """SqliteSaver at data/agent_checkpoints.db with WAL, selected by CHECKPOINT_BACKEND
    (sqlite today; switching to PostgresSaver later is a one-line change here). Returns
    None on any failure so the graph still runs without persistence."""
    backend = os.getenv("CHECKPOINT_BACKEND", "sqlite").strip().lower()
    if backend != "sqlite":
        logger.info("CHECKPOINT_BACKEND=%s not wired yet; running without checkpoints", backend)
        return None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    except Exception as exc:
        logger.info("checkpointer unavailable (%s); running without persistence", exc)
        return None


# ----------------------------------------------------------------------
# Shared state. `items`/`warnings` use add-reducers so the two fetcher nodes
# can write them concurrently and LangGraph merges (instead of clobbering).
# ----------------------------------------------------------------------
class AgentGraphState(TypedDict, total=False):
    task: str
    brief: str
    conversation: str
    items: Annotated[List[str], operator.add]     # evidence lines, merged from both fetchers
    warnings: Annotated[List[str], operator.add]
    code: str
    last_code: str
    run_result: Dict[str, Any]
    verdict: Dict[str, Any]
    score: int
    relevant: bool
    round: int
    best_score: int
    best_code: str
    best_output: str
    answer: str
    done: bool


def _rr_to_dict(r: RunResult) -> Dict[str, Any]:
    return {"ok": r.ok, "exit_code": r.exit_code, "stdout": r.stdout,
            "stderr": r.stderr, "seconds": r.seconds, "error": r.error, "summary": r.summary}


def _rr_from_dict(d: Dict[str, Any]) -> RunResult:
    return RunResult(bool(d.get("ok")), int(d.get("exit_code", -1)), d.get("stdout", ""),
                     d.get("stderr", ""), float(d.get("seconds", 0.0)), d.get("error", ""))


def _emit(on_event: OnEvent, event: Dict[str, Any]) -> None:
    if on_event:
        try:
            on_event(event)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------
def _build_graph(provider, on_event: OnEvent, budget: _Budget):
    from langgraph.graph import StateGraph, START, END

    def planner(state: AgentGraphState) -> Dict[str, Any]:
        rnd = (state.get("round") or 0) + 1
        _emit(on_event, {"type": "think", "iteration": rnd,
                         "message": f"Planning the solution (attempt {rnd}/{budget.max_rounds})…"})
        return {"round": rnd, "queries": [state["task"]]}

    def fetcher_local(state: AgentGraphState) -> Dict[str, Any]:
        from webapp.chat_logic import local_rag_enabled
        if not local_rag_enabled():
            return {"items": []}
        try:
            from backend.retrieval.hybrid_retrieve import hybrid_retrieve
            rows = hybrid_retrieve(state["task"], top_k=6) or []
        except Exception as exc:
            return {"items": [], "warnings": [f"Local retrieval skipped: {type(exc).__name__}"]}
        lines = [f"- {r.get('title', 'source')}: {(r.get('text', '') or '').strip()[:300]}"
                 for r in rows if (r.get('text') or '').strip()]
        return {"items": lines}

    def fetcher_external(state: AgentGraphState) -> Dict[str, Any]:
        try:
            from backend.external_search import gather_external_evidence, is_web_search_enabled
            if not is_web_search_enabled():
                return {"items": []}
            sources, warns = gather_external_evidence(state["task"], max_results=6)
        except Exception as exc:
            return {"items": [], "warnings": [f"External search skipped: {type(exc).__name__}"]}
        lines = []
        for s in sources[:6]:
            snippet = (getattr(s, "text", "") or getattr(s, "snippet", "") or "").strip().replace("\n", " ")
            if snippet:
                lines.append(f"- {getattr(s, 'title', 'source')}: {snippet[:300]}")
        return {"items": lines, "warnings": list(warns or [])}

    def coder(state: AgentGraphState) -> Dict[str, Any]:
        context = "\n".join(dict.fromkeys(state.get("items") or []))     # dedup, keep order
        brief = _build_brief(state["task"], state.get("brief", ""), context, state.get("conversation", ""))
        code = _generate_code(provider, brief, state.get("last_code", ""), "")
        budget.charge(brief, code)
        _emit(on_event, {"type": "code", "iteration": state.get("round", 1), "code": code})
        return {"code": code, "last_code": code}

    def sandbox_runner(state: AgentGraphState) -> Dict[str, Any]:
        # coder -> sandbox_runner is strictly sequential (correctness first). Sandbox
        # limits are whatever run_python enforces — untouched here.
        code = state.get("code", "")
        gate = pre_run(code, task=state["task"])
        if not gate.allowed:
            _emit(on_event, {"type": "blocked", "iteration": state.get("round", 1), "reason": gate.reason})
            result = RunResult(False, -1, "", "", 0.0, f"blocked by policy: {gate.reason}")
        else:
            _emit(on_event, {"type": "run", "iteration": state.get("round", 1),
                             "message": "Running it in the Docker sandbox…"})
            result = run_python(code)
        _emit(on_event, {"type": "run_result", "iteration": state.get("round", 1), "ok": result.ok,
                         "summary": result.summary, "stdout": result.stdout,
                         "stderr": result.stderr, "error": result.error})
        return {"run_result": _rr_to_dict(result)}

    def verifier(state: AgentGraphState) -> Dict[str, Any]:
        # The three checks — relevance, citation/grounding, and code-run — run CONCURRENTLY,
        # each wrapping an existing function (reviewer.review/is_relevant, verify_answer, the
        # sandbox RunResult). No threshold is touched here.
        code = state.get("code", "")
        result = _rr_from_dict(state.get("run_result") or {})
        context = "\n".join(state.get("items") or [])

        def _relevance():            # reviewer.review + is_relevant
            rev = review(code, task=state["task"])
            return {"relevant": is_relevant(rev), "summary": rev.get("summary", ""),
                    "suggestions": rev.get("suggestions", [])}

        def _citation():             # agentic_answer.verify_answer (grounding vs evidence)
            return verify_answer(provider, question=state["task"], evidence=context,
                                 answer=code, run_info=state.get("run_result"))

        def _coderun():              # the sandbox result (instant, no LLM)
            return {"ok": result.ok, "summary": result.summary}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f_rel, f_cit, f_run = ex.submit(_relevance), ex.submit(_citation), ex.submit(_coderun)
            rel = f_rel.result() or {}
            cit = f_cit.result() or {}
            run = f_run.result() or {}
        budget.charge(rel.get("summary", ""), cit.get("feedback", ""))
        verdict = {
            "relevant": rel.get("relevant", True),
            "score": int(cit.get("score", 0)),
            "ok": bool(cit.get("ok")) and bool(run.get("ok")),
            "code_ok": bool(run.get("ok")),
            "answer": rel.get("summary", ""),
            "feedback": "; ".join((rel.get("suggestions") or [])[:2]) or cit.get("feedback", ""),
        }
        _emit(on_event, {"type": "reflect", "iteration": state.get("round", 1), "verdict": verdict})
        return {"verdict": verdict, "score": verdict["score"], "relevant": bool(verdict["relevant"])}

    def grader(state: AgentGraphState) -> Dict[str, Any]:
        verdict = state.get("verdict") or {}
        score = int(state.get("score", 0))
        relevant = bool(state.get("relevant", True))
        result = _rr_from_dict(state.get("run_result") or {})
        # Best-so-far (off-topic never wins).
        best_score = int(state.get("best_score", -1))
        update: Dict[str, Any] = {}
        if relevant and result.ok and score > best_score:
            update.update(best_score=score, best_code=state.get("code", ""),
                          best_output=result.stdout, answer=verdict.get("answer", ""))
        # Pass/stop decision — thresholds used EXACTLY as configured.
        passed = verification_passed({"ok": verdict.get("ok"), "score": score}) and relevant and result.ok
        rnd = int(state.get("round", 1))
        stop = passed or rnd >= budget.max_rounds or budget.tokens_exhausted()
        if not passed and rnd >= budget.max_rounds:
            logger.info("grader: round cap %d reached (score=%d, min=%d)", budget.max_rounds, score, budget.min_score)
        update["done"] = bool(stop)
        return update

    def route(state: AgentGraphState) -> str:
        return "planner" if not state.get("done") else "END"

    g = StateGraph(AgentGraphState)
    for name, fn in (("planner", planner), ("fetcher_local", fetcher_local),
                     ("fetcher_external", fetcher_external), ("coder", coder),
                     ("sandbox_runner", sandbox_runner), ("verifier", verifier), ("grader", grader)):
        g.add_node(name, _guard(fn, name, budget))
    g.add_edge(START, "planner")
    g.add_edge("planner", "fetcher_local")          # fan-out: both fetchers run concurrently
    g.add_edge("planner", "fetcher_external")
    g.add_edge("fetcher_local", "coder")            # fan-in: coder waits for both
    g.add_edge("fetcher_external", "coder")
    g.add_edge("coder", "sandbox_runner")           # strictly sequential
    g.add_edge("sandbox_runner", "verifier")
    g.add_edge("verifier", "grader")
    g.add_conditional_edges("grader", route, {"planner": "planner", "END": END})
    return g


def _guard(fn: Callable, name: str, budget: _Budget) -> Callable:
    """Per-node wrapper: enforces the node timeout (#5), logs duration (#6), and turns any
    node error into an empty update so the graph degrades instead of crashing."""
    def wrapped(state: AgentGraphState) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                result = ex.submit(fn, state).result(timeout=budget.node_timeout) or {}
        except concurrent.futures.TimeoutError:
            logger.warning("node %s exceeded AGENT_NODE_TIMEOUT=%.0fs — skipped", name, budget.node_timeout)
        except Exception as exc:
            logger.warning("node %s failed: %s", name, exc)
        logger.info("node %s: %.1fs", name, time.time() - start)
        return result
    return wrapped


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def run_agent_graph(task: str = "", *, brief: str = "", conversation: str = "",
                    task_id: str = "", on_event: OnEvent = None) -> AgentResult:
    """Run the multi-agent graph and return an AgentResult (same shape as run_agent).
    Falls back to the hand-rolled run_agent loop if langgraph is unavailable."""
    emit: OnEvent = on_event or (lambda e: None)
    task = (task or "").strip()
    if not task:
        return AgentResult(task, False, "", "", "No task given.", [])
    if not graph_available():
        return run_agent(task, brief=brief, conversation=conversation, on_event=on_event)

    provider = get_provider(os.getenv("AGENT_MODEL") or None)
    if not provider.is_available:
        emit({"type": "error", "message": "LLM not available."})
        return AgentResult(task, False, "", "", "LLM unavailable.", [])
    if not docker_available():
        emit({"type": "error", "message": "Docker is not running — start Docker Desktop."})
        return AgentResult(task, False, "", "", "Docker unavailable.", [])

    budget = _Budget()
    started = time.time()
    saver = _make_checkpointer()
    compiled = _build_graph(provider, emit, budget).compile(checkpointer=saver) if saver \
        else _build_graph(provider, emit, budget).compile()
    config = {"configurable": {"thread_id": task_id or uuid.uuid4().hex},
              "recursion_limit": 8 * max(1, budget.max_rounds) + 10}
    try:
        final = compiled.invoke({"task": task, "brief": brief, "conversation": conversation}, config=config)
    except Exception as exc:
        logger.warning("graph invoke failed (%s); falling back to run_agent", exc)
        return run_agent(task, brief=brief, conversation=conversation, on_event=on_event)

    res = AgentResult(
        task=task,
        success=bool(final.get("best_code") and int(final.get("best_score", 0)) >= budget.min_score),
        best_code=final.get("best_code", ""),
        best_output=final.get("best_output", ""),
        answer=final.get("answer", ""),
        attempts=[],
    )
    logger.info("run_agent_graph: %d round(s), %.1fs total, ~%d tokens",
                final.get("round", 0), time.time() - started, budget.tokens_used)
    emit({"type": "final", "success": res.success, "answer": res.answer,
          "code": res.best_code, "output": res.best_output, "iterations": final.get("round", 0)})
    return res
