"""
Autonomous deep-research agent.

Give it ONE research question and it does the whole job on its own and returns a
finished, cited report — no manual steps:

    PLAN     -> break the question into focused sub-questions
    EXECUTE  -> search EVERYWHERE for each (web, arXiv, Semantic Scholar,
                Wikipedia, patents, GitHub), accumulating de-duplicated evidence
    REFLECT  -> judge what is still missing; spawn follow-up searches
    (loop the three above until the question is well covered or the round budget ends)
    WRITE    -> compose a comprehensive, well-structured, cited report
    REVIEW   -> self-critique with the peer reviewer; one revision pass if needed

Run it:

    python -m backend.agent.research_agent "How do modern neural beamformers compare to MVDR?"
    python -m backend.agent.research_agent --rounds 2 --per-query 6 "your question"

Design credit (ideas only, original code): the THINK->EXECUTE->REFLECT loop and the
bounded two-tier memory come from auto-deep-researcher-24x7 (Apache-2.0); the
self-review stage from the Awesome-AI-Scientist survey's "review systems".
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.agent.memory import TwoTierMemory
from backend.answering.reviewer import review
from backend.external_search.orchestrator import gather_external_evidence
from backend.llm.streaming_provider import get_provider

MAX_ROUNDS = int(os.getenv("RESEARCH_MAX_ROUNDS", "3"))
PER_QUERY_RESULTS = int(os.getenv("RESEARCH_PER_QUERY_RESULTS", "8"))
MAX_SUBQUESTIONS = int(os.getenv("RESEARCH_MAX_SUBQUESTIONS", "5"))
EVIDENCE_CHARS_PER_SOURCE = int(os.getenv("RESEARCH_EVIDENCE_CHARS", "1400"))
MAX_EVIDENCE_SOURCES = int(os.getenv("RESEARCH_MAX_EVIDENCE_SOURCES", "28"))
REPORT_MAX_TOKENS = int(os.getenv("RESEARCH_REPORT_MAX_TOKENS", "4000"))
STEP_MAX_TOKENS = int(os.getenv("RESEARCH_STEP_MAX_TOKENS", "1500"))

OnEvent = Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class ResearchReport:
    question: str
    report: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    rounds: int = 0
    review: Optional[Dict[str, Any]] = None
    error: str = ""


# ----------------------------------------------------------------------
# Small LLM + parsing helpers
# ----------------------------------------------------------------------
def _complete(provider, system: str, user: str, max_tokens: int, temperature: float = 0.2) -> str:
    return "".join(provider.stream_chat(
        [{"role": "user", "content": user}], system=system,
        max_tokens=max_tokens, temperature=temperature)).strip()


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else {}
        except Exception:
            pass
    return {}


def _emit(on_event: OnEvent, event: Dict[str, Any]) -> None:
    if on_event:
        try:
            on_event(event)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Evidence handling (search EVERYWHERE, de-dup, number for citations)
# ----------------------------------------------------------------------
def _to_item(es: Any) -> Dict[str, Any]:
    d = es.to_public() if hasattr(es, "to_public") else dict(es)
    d["text"] = (getattr(es, "text", "") or getattr(es, "snippet", "") or d.get("text", "")).strip()
    return d


def _item_key(it: Dict[str, Any]) -> str:
    return ((it.get("url") or "").strip().lower().rstrip("/")
            or (it.get("title") or "").strip().lower())


def _search(query: str, k: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        sources, warnings = gather_external_evidence(query, max_results=k)
    except Exception as exc:
        return [], [f"search failed: {exc}"]
    return [_to_item(s) for s in sources], warnings


def _format_evidence(items: List[Dict[str, Any]]) -> str:
    lines = []
    for i, it in enumerate(items[:MAX_EVIDENCE_SOURCES], 1):
        kind = (it.get("source_type") or "web").replace("_", " ")
        title = it.get("title") or "Untitled"
        url = it.get("url") or it.get("file_path") or ""
        text = (it.get("text") or "")[:EVIDENCE_CHARS_PER_SOURCE]
        lines.append(f"[{i}] ({kind}) {title} -- {url}\n{text}")
    return "\n\n".join(lines)


def _public_sources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, it in enumerate(items[:MAX_EVIDENCE_SOURCES], 1):
        pub = dict(it)
        pub["n"] = i
        pub["text"] = (it.get("text") or "")[:600]
        out.append(pub)
    return out


# ----------------------------------------------------------------------
# The four reasoning steps
# ----------------------------------------------------------------------
_PLAN_SYSTEM = (
    "You are a research planner. Break the user's question into a few focused, "
    "non-overlapping sub-questions that together fully cover it (definitions, how it "
    "works, comparisons, evidence, current state). Reply with ONLY JSON: "
    '{"subquestions": ["...", "..."]}'
)

_SYNTH_SYSTEM = (
    "You are a careful research analyst. Using ONLY the numbered evidence, write tight, "
    "factual findings (bullet points) that address the sub-questions. Cite every point "
    "with [n] matching the evidence. Note conflicts and recency. Never invent facts."
)

_REFLECT_SYSTEM = (
    "You are a rigorous research critic. Given the question and the findings gathered so "
    "far, decide whether the research now COMPREHENSIVELY answers the question. Reply with "
    "ONLY JSON: {\"done\": true|false, \"gaps\": [\"what is still missing\"], "
    "\"next_queries\": [\"specific search queries to fill the gaps\"]}. "
    "Set done=true only if little of value remains to find."
)

_REPORT_SYSTEM = (
    "You are an expert research writer. Using ONLY the numbered evidence, write a "
    "comprehensive, well-structured research report that answers the question.\n"
    "- Open with a direct 2-4 sentence summary, then sections with markdown headings.\n"
    "- SYNTHESIZE across sources; compare findings; note agreements, disagreements, and "
    "recency (mention dates when known).\n"
    "- Cite every non-trivial claim with [n], drawing on a DIVERSITY of sources.\n"
    "- Ground all specifics in the evidence; never invent facts, numbers, or citations.\n"
    "- End with a short '## Key takeaways' list.\n"
    "- Prefer depth and breadth over brevity."
)


def _plan(provider, question: str) -> List[str]:
    raw = _complete(provider, _PLAN_SYSTEM, question, max_tokens=STEP_MAX_TOKENS)
    subs = _parse_json(raw).get("subquestions") or []
    subs = [str(s).strip() for s in subs if str(s).strip()]
    return subs[:MAX_SUBQUESTIONS] or [question]


def _synthesize(provider, question: str, sub_qs: List[str], evidence: str, progress: str) -> str:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"SUB-QUESTIONS THIS ROUND:\n- " + "\n- ".join(sub_qs) + "\n\n"
        f"NUMBERED EVIDENCE:\n{evidence}\n\n"
        f"{('PROGRESS SO FAR:' + chr(10) + progress + chr(10) + chr(10)) if progress else ''}"
        "Write the findings now."
    )
    return _complete(provider, _SYNTH_SYSTEM, user, max_tokens=STEP_MAX_TOKENS)


def _reflect(provider, question: str, findings: List[str]) -> Dict[str, Any]:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"FINDINGS GATHERED SO FAR:\n{chr(10).join(findings)}\n\n"
        "Is the question comprehensively answered? Reply with the JSON."
    )
    verdict = _parse_json(_complete(provider, _REFLECT_SYSTEM, user, max_tokens=STEP_MAX_TOKENS))
    verdict.setdefault("done", False)
    verdict["done"] = bool(verdict.get("done"))
    nq = verdict.get("next_queries") or []
    verdict["next_queries"] = [str(x).strip() for x in nq if str(x).strip()][:MAX_SUBQUESTIONS]
    return verdict


def _write_report(provider, question: str, evidence: str, findings: List[str]) -> str:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"RESEARCH NOTES (your own findings, already cited):\n{chr(10).join(findings)}\n\n"
        f"FULL NUMBERED EVIDENCE (cite these as [n]):\n{evidence}\n\n"
        "Write the complete report now."
    )
    return _complete(provider, _REPORT_SYSTEM, user, max_tokens=REPORT_MAX_TOKENS, temperature=0.3)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def research(
    question: str,
    *,
    max_rounds: int = MAX_ROUNDS,
    per_query_results: int = PER_QUERY_RESULTS,
    do_review: bool = True,
    on_event: OnEvent = None,
) -> ResearchReport:
    """Run the full autonomous research loop and return a finished report."""
    question = (question or "").strip()
    if not question:
        return ResearchReport(question="", report="", error="No question given.")

    # Optional: orchestrate the same pipeline with LangGraph (RESEARCH_ENGINE=langgraph).
    if (os.getenv("RESEARCH_ENGINE", "").strip().lower() == "langgraph"):
        try:
            from backend.agent.langgraph_research import run as _lg_run, langgraph_available
            if langgraph_available():
                return _lg_run(question, max_rounds=max_rounds,
                               per_query_results=per_query_results, on_event=on_event)
            _emit(on_event, {"type": "warning",
                             "message": "RESEARCH_ENGINE=langgraph but langgraph is not installed; using the built-in engine."})
        except Exception as exc:  # never let the optional engine break research
            _emit(on_event, {"type": "warning", "message": f"LangGraph engine unavailable ({exc}); using built-in."})

    provider = get_provider()
    if not provider.is_available:
        return ResearchReport(question=question, report="",
                              error=provider.unavailable_message())

    _emit(on_event, {"type": "status", "message": "Planning the research..."})
    sub_qs = _plan(provider, question)
    _emit(on_event, {"type": "plan", "subquestions": sub_qs})

    mem = TwoTierMemory(brief=f"Research goal: {question}\nSub-questions:\n- " + "\n- ".join(sub_qs))
    all_items: List[Dict[str, Any]] = []
    seen = set()
    findings: List[str] = []
    open_queries = list(sub_qs)
    rounds_done = 0

    for round_no in range(1, max_rounds + 1):
        rounds_done = round_no
        _emit(on_event, {"type": "round", "round": round_no, "of": max_rounds})

        # EXECUTE: search everywhere for each open query
        round_items: List[Dict[str, Any]] = []
        for q in open_queries:
            _emit(on_event, {"type": "search", "query": q})
            items, warnings = _search(q, per_query_results)
            for w in warnings:
                _emit(on_event, {"type": "warning", "message": w})
            for it in items:
                key = _item_key(it)
                if key and key not in seen:
                    seen.add(key)
                    all_items.append(it)
                    round_items.append(it)
        _emit(on_event, {"type": "sources", "found": len(round_items), "total": len(all_items)})

        # SYNTHESIZE this round's evidence into findings
        _emit(on_event, {"type": "status", "message": f"Synthesizing round {round_no}..."})
        note = _synthesize(provider, question, open_queries,
                           _format_evidence(round_items or all_items), mem.context())
        if note:
            findings.append(f"### Round {round_no} findings\n{note}")
            mem.append(f"Round {round_no}: {note[:300]}")

        # REFLECT: decide whether to stop or search more
        if round_no >= max_rounds:
            break
        verdict = _reflect(provider, question, findings)
        _emit(on_event, {"type": "reflect", "done": verdict["done"],
                         "gaps": verdict.get("gaps", []), "next": verdict.get("next_queries", [])})
        if verdict["done"] or not verdict["next_queries"]:
            break
        open_queries = verdict["next_queries"]

    # WRITE the final report
    _emit(on_event, {"type": "status", "message": "Writing the final report..."})
    report = _write_report(provider, question, _format_evidence(all_items), findings)

    # REVIEW (self-critique) + optional single revision
    rev = None
    if do_review and report:
        _emit(on_event, {"type": "status", "message": "Self-reviewing the report..."})
        rev = review(report)
        if rev and not rev.get("error") and rev.get("recommendation") in ("major revision", "reject"):
            _emit(on_event, {"type": "status", "message": "Revising after review..."})
            fixes = "; ".join((rev.get("weaknesses") or []) + (rev.get("suggestions") or []))[:1500]
            report = _write_report(
                provider, question, _format_evidence(all_items),
                findings + [f"### Reviewer asked you to fix\n{fixes}"])

    pub_sources = _public_sources(all_items)
    # Append a self-contained Sources list so the [n] citations stay readable.
    if report and pub_sources:
        refs = "\n".join(
            f"[{s['n']}] {(s.get('title') or 'source')} — {s.get('url') or s.get('file_path') or ''}".rstrip(" —")
            for s in pub_sources)
        report = report.rstrip() + "\n\n## Sources\n" + refs

    out = ResearchReport(
        question=question, report=report, sources=pub_sources,
        sub_questions=sub_qs, rounds=rounds_done, review=rev)
    _emit(on_event, {"type": "done", "report": report, "rounds": rounds_done,
                     "sources": len(pub_sources)})
    return out


# ----------------------------------------------------------------------
def _main() -> int:
    import argparse
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Autonomous deep-research agent — give a question, get a cited report.")
    ap.add_argument("question", nargs="*", help="The research question.")
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="Max search/reflect rounds.")
    ap.add_argument("--per-query", type=int, default=PER_QUERY_RESULTS, help="Results per search query.")
    ap.add_argument("--no-review", action="store_true", help="Skip the self-review pass.")
    args = ap.parse_args()

    question = " ".join(args.question).strip() or (sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not question:
        print("Give a research question. Example:\n  python -m backend.agent.research_agent \"...\"")
        return 2

    def log(ev: Dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "plan":
            print("Plan:\n  - " + "\n  - ".join(ev["subquestions"]) + "\n")
        elif t == "round":
            print(f"--- Round {ev['round']}/{ev['of']} ---")
        elif t == "search":
            print(f"  search: {ev['query']}")
        elif t == "sources":
            print(f"  evidence: +{ev['found']} (total {ev['total']})")
        elif t == "reflect":
            print(f"  reflect: done={ev['done']} gaps={len(ev.get('gaps', []))}")
        elif t == "status":
            print(ev["message"])

    res = research(question, max_rounds=args.rounds, per_query_results=args.per_query,
                   do_review=not args.no_review, on_event=log)
    if res.error:
        print("Error:", res.error)
        return 1
    print("\n" + "=" * 70 + "\nREPORT\n" + "=" * 70 + "\n")
    print(res.report)
    print(f"\n[{len(res.sources)} sources, {res.rounds} round(s)", end="")
    if res.review and not res.review.get("error"):
        print(f", review: {res.review.get('recommendation', '?')}]", end="")
    print("]" if not (res.review and not res.review.get("error")) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
