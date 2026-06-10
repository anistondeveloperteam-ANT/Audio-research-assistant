"""
LangGraph-orchestrated deep research.

The same PLAN -> SEARCH -> SYNTHESIZE -> REFLECT (loop) -> REPORT pipeline as
`research_agent.research`, but expressed as an explicit LangGraph state machine.
The reasoning *steps* are the proven functions from `research_agent` (reused, not
re-implemented); LangGraph only provides the graph: typed shared state, named
nodes, and a conditional edge that loops back to SEARCH until the question is
covered or the round budget runs out.

    PLAN ──> SEARCH ──> SYNTHESIZE ──> REFLECT ──(needs more?)──> SEARCH
                                          │
                                          └────(done)──────────> REPORT ──> END

Why LangGraph: the loop, the shared state, and the "search again vs. finish"
branch become declarative and inspectable (and gain checkpointing/streaming for
free) instead of living in a hand-rolled for-loop.

Run it:

    python -m backend.agent.langgraph_research "How do neural beamformers compare to MVDR?"

Or make the web app use it for /api/research by setting RESEARCH_ENGINE=langgraph.
LangGraph is an optional dependency: if it isn't installed, callers fall back to
`research_agent.research`.
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any, Callable, Dict, List, Optional, TypedDict

from backend.agent import research_agent as ra
from backend.llm.streaming_provider import get_provider

OnEvent = Optional[Callable[[Dict[str, Any]], None]]


def langgraph_available() -> bool:
    """True if the optional `langgraph` package is importable."""
    return importlib.util.find_spec("langgraph") is not None


class ResearchState(TypedDict, total=False):
    """Shared state passed between graph nodes. Each node returns a partial update
    (returned keys REPLACE the value in state — we read-then-return the full list)."""
    question: str
    sub_questions: List[str]
    open_queries: List[str]      # queries to run in the next SEARCH
    items: List[Dict[str, Any]]  # de-duplicated evidence gathered so far
    seen: List[str]              # item keys already collected (for dedup)
    findings: List[str]          # per-round synthesized notes
    round: int
    max_rounds: int
    per_query: int
    done: bool
    report: str


def _build_graph(provider, on_event: OnEvent):
    """Compile the research StateGraph. Nodes close over the provider + event sink
    so the shared state stays plain data."""
    from langgraph.graph import StateGraph, START, END

    def plan(state: ResearchState) -> Dict[str, Any]:
        ra._emit(on_event, {"type": "status", "message": "Planning the research..."})
        subs = ra._plan(provider, state["question"])
        ra._emit(on_event, {"type": "plan", "subquestions": subs})
        return {"sub_questions": subs, "open_queries": subs,
                "items": [], "seen": [], "findings": [], "round": 0, "done": False}

    def search(state: ResearchState) -> Dict[str, Any]:
        items = list(state.get("items") or [])
        seen = set(state.get("seen") or [])
        per_q = state.get("per_query") or ra.PER_QUERY_RESULTS
        found_this_round = 0
        for q in (state.get("open_queries") or [state["question"]]):
            ra._emit(on_event, {"type": "search", "query": q})
            results, warnings = ra._search(q, per_q)
            for w in warnings:
                ra._emit(on_event, {"type": "warning", "message": w})
            for it in results:
                key = ra._item_key(it)
                if key and key not in seen:
                    seen.add(key)
                    items.append(it)
                    found_this_round += 1
        ra._emit(on_event, {"type": "sources", "found": found_this_round, "total": len(items)})
        return {"items": items, "seen": sorted(seen)}

    def synthesize(state: ResearchState) -> Dict[str, Any]:
        rnd = (state.get("round") or 0) + 1
        ra._emit(on_event, {"type": "status", "message": f"Synthesizing round {rnd}..."})
        findings = list(state.get("findings") or [])
        note = ra._synthesize(provider, state["question"], state.get("open_queries") or [],
                              ra._format_evidence(state.get("items") or []), "\n".join(findings))
        if note:
            findings.append(f"### Round {rnd} findings\n{note}")
        return {"findings": findings}

    def reflect(state: ResearchState) -> Dict[str, Any]:
        rnd = (state.get("round") or 0) + 1
        max_rounds = state.get("max_rounds") or ra.MAX_ROUNDS
        if rnd >= max_rounds:
            return {"round": rnd, "done": True, "open_queries": []}
        verdict = ra._reflect(provider, state["question"], state.get("findings") or [])
        done = bool(verdict.get("done"))
        ra._emit(on_event, {"type": "reflect", "done": done, "round": rnd})
        return {"round": rnd, "done": done, "open_queries": verdict.get("next_queries") or []}

    def report(state: ResearchState) -> Dict[str, Any]:
        ra._emit(on_event, {"type": "status", "message": "Writing the report..."})
        text = ra._write_report(provider, state["question"],
                                ra._format_evidence(state.get("items") or []),
                                state.get("findings") or [])
        return {"report": text}

    def route(state: ResearchState) -> str:
        return "search" if (not state.get("done") and state.get("open_queries")) else "report"

    g = StateGraph(ResearchState)
    g.add_node("plan", plan)
    g.add_node("search", search)
    g.add_node("synthesize", synthesize)
    g.add_node("reflect", reflect)
    g.add_node("report", report)
    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", "reflect")
    g.add_conditional_edges("reflect", route, {"search": "search", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


def run(
    question: str,
    *,
    max_rounds: int = ra.MAX_ROUNDS,
    per_query_results: int = ra.PER_QUERY_RESULTS,
    on_event: OnEvent = None,
) -> "ra.ResearchReport":
    """Run the LangGraph research pipeline and return a finished `ResearchReport`."""
    question = (question or "").strip()
    if not question:
        return ra.ResearchReport(question="", report="", error="No question given.")
    if not langgraph_available():
        # Graceful fallback so the feature degrades instead of crashing.
        return ra.research(question, max_rounds=max_rounds,
                           per_query_results=per_query_results, on_event=on_event)

    provider = get_provider()
    if not provider.is_available:
        return ra.ResearchReport(question=question, report="",
                                 error=provider.unavailable_message())

    graph = _build_graph(provider, on_event)
    final = graph.invoke(
        {"question": question, "max_rounds": max_rounds, "per_query": per_query_results},
        config={"recursion_limit": 4 * max(1, max_rounds) + 6},
    )
    return ra.ResearchReport(
        question=question,
        report=final.get("report", ""),
        sources=ra._public_sources(final.get("items") or []),
        sub_questions=final.get("sub_questions") or [],
        rounds=final.get("round", 0),
    )


def _main() -> int:
    args = [a for a in sys.argv[1:] if a]
    if not args:
        print('Usage: python -m backend.agent.langgraph_research "your question"')
        return 2
    if not langgraph_available():
        print("langgraph is not installed. Run: pip install langgraph", file=sys.stderr)
        return 1
    report = run(" ".join(args), on_event=lambda e: print(f"  [{e.get('type')}] "
                 f"{e.get('message') or e.get('query') or e.get('subquestions') or ''}"))
    print("\n" + "=" * 70 + "\n" + (report.report or report.error) + "\n")
    print(f"({report.rounds} rounds, {len(report.sources)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
