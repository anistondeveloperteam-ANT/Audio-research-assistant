"""
Server-side chat orchestration for the web UI.

Reuses the existing backend (retrieval, LLM, memory) and yields a stream of
small JSON events that the browser renders. No backend code is modified here;
this module only wires the proven pieces together for the new UI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env BEFORE reading the settings below (this module may be imported before
# anything else triggers dotenv, so the flags must not read a stale environment).
try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from backend.memory.store import MemoryStore, default_db_path  # noqa: E402
from backend.answering.query_sanity import check_query_sanity  # noqa: E402
from backend.answering.agentic_answer import (  # noqa: E402
    agentic_loop_enabled,
    build_revision_message,
    complete_text,
    followup_query,
    max_verify_rounds,
    run_best_python_block,
    verification_footer,
    verification_passed,
    verify_answer,
)
from backend.llm.streaming_provider import get_provider  # noqa: E402
from backend.external_search import gather_external_evidence, is_web_search_enabled  # noqa: E402


def local_rag_enabled() -> bool:
    """True when the optional local PDF RAG (Oracle + embeddings) is turned on.
    Read live so it always reflects the current .env."""
    return (os.getenv("ENABLE_LOCAL_RAG", "false") or "").strip().lower() in ("1", "true", "yes", "on")


# Back-compat constant (read once after .env is loaded).
ENABLE_LOCAL_RAG = local_rag_enabled()

# ----------------------------------------------------------------------
# Singletons
# ----------------------------------------------------------------------
_memory: MemoryStore | None = None


def memory() -> MemoryStore:
    global _memory
    if _memory is None:
        _memory = MemoryStore(default_db_path(ROOT))
    return _memory


# ----------------------------------------------------------------------
# Prompt helpers
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert research assistant for audio signal processing and audio AI.\n"
    "Answer using ONLY the numbered source excerpts in the user's message. Each source\n"
    "is tagged with its type: (paper) = the user's local papers, (web) = a web page,\n"
    "(github) = a public repository file, (pdf) = an online PDF.\n"
    "- Be precise, technical, and clear; prefer short paragraphs and bullet points.\n"
    "- Cite every non-trivial claim with [1], [2], ... matching the numbered sources.\n"
    "- Prefer the local (paper) sources for project-specific answers; use (web)/(pdf)\n"
    "  sources for latest/current information.\n"
    "- If external sources conflict with the local papers, say so explicitly.\n"
    "- If the sources do not address the question, say so plainly; never invent facts,\n"
    "  numbers, URLs, or titles.\n"
    "- For code / implementation / simulation requests: read the method from the\n"
    "  cited paper/source, then write COMPLETE, RUNNABLE, ORIGINAL code (imports +\n"
    "  a small example / simulation, e.g. NumPy/SciPy) — explain the steps and cite\n"
    "  the source for each. Do NOT copy code verbatim from repositories; reimplement\n"
    "  the idea in this project's style and note any license constraints.\n"
    "- Prefer depth and accuracy over brevity; ground specifics (equations, numbers,\n"
    "  parameters) in the cited sources.\n"
)


def _evidence_header(n: int, item: Dict[str, Any]) -> str:
    """Build the '[n] (type) title — location' header for one evidence item."""
    st = item.get("source_type", "local_pdf")
    title = item.get("title") or "Untitled"
    if st == "local_pdf":
        section = item.get("section") or item.get("section_name") or "?"
        ps = item.get("page_start") or "?"
        pe = item.get("page_end") or "?"
        return f"[{n}] (paper) {title} -- {section} (pages {ps}-{pe})"
    if st in ("github_repo", "github_code"):
        loc = item.get("file_path") or ""
        if item.get("line_start"):
            loc += f":{item['line_start']}" + (f"-{item['line_end']}" if item.get("line_end") else "")
        lic = f" [license: {item['license']}]" if item.get("license") else ""
        return f"[{n}] (github) {title} -- {item.get('url', '')} {loc}{lic}".rstrip()
    if st == "online_pdf":
        pg = f" p.{item['page']}" if item.get("page") else ""
        return f"[{n}] (pdf) {title} -- {item.get('url', '')}{pg}"
    return f"[{n}] (web) {title} -- {item.get('url', '')}"


# How much of each source's text the model actually reads. Bigger = more accurate,
# deeper answers (and enough method detail to write code), at higher token cost.
EVIDENCE_CHARS_PER_SOURCE = int(os.getenv("EVIDENCE_CHARS_PER_SOURCE", "3500"))


def format_evidence(sources: List[Dict[str, Any]], max_chars: int = EVIDENCE_CHARS_PER_SOURCE) -> str:
    """Format local and/or external evidence items into a numbered, cited block.
    Works on raw local retrieval dicts (treated as papers) and on external dicts
    that carry a `source_type`."""
    if not sources:
        return "(no retrieved sources)"
    parts = []
    for i, r in enumerate(sources, 1):
        text = (r.get("text") or r.get("chunk_text") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        parts.append(_evidence_header(i, r) + "\n" + text)
    return "\n\n".join(parts)


def build_user_message(question: str, evidence: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Retrieved evidence (your local papers and any external sources):\n\n{evidence}\n\n"
        f"Answer the question using only the evidence above. Cite sources with [n]."
    )


# Sections that rarely contain real answers — drop them from the evidence so the
# shown/used sources stay relevant (References lists, acknowledgements, etc.).
_LOW_VALUE_SECTIONS = ("reference", "bibliograph", "acknowledg", "author contribution",
                       "funding", "conflict of interest", "appendix")

# Adaptive source count: keep every chunk whose relevance (reranker score) clears
# the threshold, bounded by [MIN, MAX]. So an easy/narrow question may return 4
# sources and a broad one 11 — the number reflects how much is actually relevant,
# instead of always being a fixed top_k. Tunable via .env.
SOURCE_MIN_SCORE = float(os.getenv("SOURCE_MIN_SCORE", "0.30"))
SOURCE_MIN = int(os.getenv("SOURCE_MIN", "3"))
SOURCE_MAX = int(os.getenv("SOURCE_MAX", "12"))

# How many external sources to keep (accuracy > brevity — keep more), and how many
# tokens the answer may use (large enough for full code / simulations).
EXTERNAL_TOP_K = int(os.getenv("EXTERNAL_TOP_K", "20"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "8000"))  # room for full code, no truncation
AGENTIC_EXTRA_SEARCH_K = int(os.getenv("AGENTIC_EXTRA_SEARCH_K", "8"))


def _score(r: Dict[str, Any]) -> float:
    try:
        return float(r.get("rerank_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop low-value sections, then keep as many *relevant* sources as clear the
    score threshold (between SOURCE_MIN and SOURCE_MAX). The count varies per
    query instead of being a fixed number."""
    def is_low_value(r: Dict[str, Any]) -> bool:
        section = (r.get("section") or r.get("section_name") or "").lower()
        return any(key in section for key in _LOW_VALUE_SECTIONS)

    kept = [r for r in results if not is_low_value(r)] or list(results)
    kept.sort(key=_score, reverse=True)

    relevant = [r for r in kept if _score(r) >= SOURCE_MIN_SCORE]
    if len(relevant) < SOURCE_MIN:        # too few cleared the bar -> keep the best anyway
        relevant = kept[:SOURCE_MIN]
    return relevant[:SOURCE_MAX]


def public_source(r: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Trim a LOCAL retrieval result down to what the UI needs to render a card."""
    return {
        "n": i,
        "source_type": "local_pdf",
        "title": r.get("title") or "Untitled",
        "section": r.get("section") or r.get("section_name") or "",
        "page_start": r.get("page_start"),
        "page_end": r.get("page_end"),
        "text": (r.get("text") or r.get("chunk_text") or "").strip()[:600],
        "score": round(float(r.get("rerank_score") or 0.0), 3),
        "retrieval_sources": r.get("retrieval_sources") or [],
        "graph_reason": r.get("graph_reason") or "",
    }


def _local_evidence_item(r: Dict[str, Any]) -> Dict[str, Any]:
    """Full-text local evidence item (used for the LLM context + UI card)."""
    return {
        "source_type": "local_pdf",
        "title": r.get("title") or "Untitled",
        "section": r.get("section") or r.get("section_name") or "",
        "page_start": r.get("page_start"),
        "page_end": r.get("page_end"),
        "url": "", "file_path": None, "line_start": None, "line_end": None, "page": None,
        "provider": "local", "license": None,
        "text": (r.get("text") or r.get("chunk_text") or "").strip(),
        "score": round(float(r.get("rerank_score") or 0.0), 3),
        "retrieval_sources": r.get("retrieval_sources") or [],
        "graph_reason": r.get("graph_reason") or "",
        "concepts": r.get("concepts") or "",
    }


def _gather_local_items(query: str, mode: str) -> tuple[List[Dict[str, Any]], List[str]]:
    """Search the optional local PDF RAG and return full-text evidence items."""
    try:
        # Imported lazily so a web-only deploy needs no Oracle / heavy ML deps.
        from backend.answering.research_modes import apply_research_mode
        from backend.retrieval.hybrid_retrieve import hybrid_retrieve

        try:
            apply_research_mode(mode)
        except Exception:
            pass
        local = select_sources(hybrid_retrieve(query, top_k=SOURCE_MAX + 6) or [])
        return [_local_evidence_item(r) for r in local], []
    except Exception as exc:
        return [], [f"Local paper search is unavailable: {exc}"]


def _external_item(es: Any) -> Dict[str, Any]:
    d = es.to_public()
    d["text"] = (getattr(es, "text", "") or getattr(es, "snippet", "") or "").strip()
    return d


def _gather_external_items(query: str, max_results: int) -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        ext_sources, warnings = gather_external_evidence(query, max_results=max_results)
    except Exception as exc:
        return [], [f"External search failed: {exc}"]
    return [_external_item(es) for es in ext_sources], warnings


def _item_key(item: Dict[str, Any]) -> tuple:
    text = (item.get("text") or "").strip().lower()[:240]
    return (
        item.get("source_type") or "",
        (item.get("url") or "").strip().lower().rstrip("/"),
        (item.get("file_path") or "").strip().lower(),
        item.get("page") or item.get("page_start") or "",
        (item.get("title") or "").strip().lower(),
        text,
    )


def _extend_unique(items: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> int:
    seen = {_item_key(it) for it in items}
    added = 0
    for it in new_items:
        key = _item_key(it)
        if key in seen:
            continue
        seen.add(key)
        items.append(it)
        added += 1
    return added


def _public_sources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    for i, it in enumerate(items, 1):
        pub = dict(it)
        pub["n"] = i
        pub["text"] = (it.get("text") or "")[:600]
        sources.append(pub)
    return sources


# ----------------------------------------------------------------------
# The streaming orchestration
# ----------------------------------------------------------------------
def stream_chat_events(
    session_id: str,
    question: str,
    mode: str = "Default",
    top_k: int = 8,
    web_search: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield event dicts: sanity | status | sources | token | warning | done | error.

    Web search is the PRIMARY knowledge source. The local Oracle/PDF RAG is
    optional and off unless ENABLE_LOCAL_RAG=true, so the app runs in production
    with no Oracle database and no uploaded papers — just a web-search key + LLM.
    """
    q = (question or "").strip()

    sanity = check_query_sanity(q)
    if not sanity.ok:
        yield {"type": "sanity", "message": sanity.user_message or "Please rephrase your question."}
        return

    mem = memory()
    mem.append_turn(session_id, "user", q)

    items: List[Dict[str, Any]] = []
    local_on = local_rag_enabled()

    # --- 1) Local PDF papers (when enabled; needs Oracle + indexed papers) ---
    if local_on:
        yield {"type": "status", "message": "Searching your papers..."}
        local_items, local_warnings = _gather_local_items(q, mode)
        _extend_unique(items, local_items)
        for w in local_warnings:
            yield {"type": "warning", "message": w}

    # --- 2) ALWAYS search everywhere too: web, research papers (arXiv + Semantic
    #        Scholar), Wikipedia, patents & GitHub — combined with the papers. ---
    if is_web_search_enabled():
        yield {"type": "status", "message": (
            "Searching your papers + the web, research papers, patents & GitHub..."
            if local_on else
            "Searching the web, research papers, patents & GitHub...")}
        ext_items, ext_warnings = _gather_external_items(q, EXTERNAL_TOP_K)
        _extend_unique(items, ext_items)
        for w in ext_warnings:
            yield {"type": "warning", "message": w}

    # --- Nothing available at all -> explain instead of guessing ---
    if not items:
        if not local_on and not is_web_search_enabled():
            msg = ("No knowledge source is enabled. Set `ENABLE_WEB_SEARCH=true` (and "
                   "optionally `TAVILY_API_KEY` for web pages & patents) in `.env`, or "
                   "turn on local papers with `ENABLE_LOCAL_RAG=true`.")
        else:
            msg = "I couldn't find relevant information for that question in the available sources."
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": msg}
        mem.append_turn(session_id, "assistant", msg, sources=[])
        yield {"type": "done", "answer": msg}
        return

    sources = _public_sources(items)
    yield {"type": "sources", "sources": sources}

    recent = mem.get_recent_turns(session_id, n_messages=6)
    # Replace the just-stored bare question with the evidence-augmented version.
    history = recent[:-1] if recent and recent[-1]["role"] == "user" else recent

    answer_parts: List[str] = []
    try:
        provider = get_provider()
        if not provider.is_available:
            note = (
                "The language model isn't available right now, so I can't write a full "
                "answer — but the most relevant sources are shown on the right."
            )
            answer_parts.append(note)
            yield {"type": "token", "text": note}
        elif agentic_loop_enabled():
            answer = ""
            verdict: Dict[str, Any] = {}
            run_info: Dict[str, Any] | None = None
            rounds_done = 0
            for round_no in range(1, max_verify_rounds() + 1):
                rounds_done = round_no
                evidence = format_evidence(items)
                if answer and verdict:
                    user_msg = build_revision_message(
                        question=q,
                        evidence=evidence,
                        previous_answer=answer,
                        verdict=verdict,
                        run_info=run_info,
                    )
                else:
                    user_msg = build_user_message(q, evidence)
                messages = history + [{"role": "user", "content": user_msg}]

                yield {"type": "status", "message": (
                    f"Agent loop {round_no}/{max_verify_rounds()}: drafting a grounded answer..."
                )}
                answer = complete_text(
                    provider,
                    messages,
                    system=SYSTEM_PROMPT,
                    max_tokens=ANSWER_MAX_TOKENS,
                    temperature=0.3,
                )

                yield {"type": "status", "message": "Checking for runnable Python simulation..."}
                run_info = run_best_python_block(answer)
                if run_info:
                    if run_info.get("attempted"):
                        yield {"type": "status", "message": f"Sandbox result: {run_info.get('summary')}"}
                    else:
                        yield {"type": "warning", "message": run_info.get("summary", "Simulation was not run.")}

                yield {"type": "status", "message": "Verifying answer against the retrieved evidence..."}
                try:
                    verdict = verify_answer(
                        provider,
                        question=q,
                        evidence=evidence,
                        answer=answer,
                        run_info=run_info,
                    )
                except Exception as exc:
                    verdict = {
                        "ok": False,
                        "score": 0,
                        "needs_more_search": False,
                        "feedback": f"Verification failed: {exc}",
                    }
                    yield {"type": "warning", "message": f"Verification failed: {exc}"}
                    break

                run_failed = bool(run_info and run_info.get("attempted") and not run_info.get("ok"))
                if run_failed and not verdict.get("feedback"):
                    verdict["feedback"] = "Generated Python did not run successfully; fix the code and rerun it."

                if (verification_passed(verdict) and not run_failed) or round_no >= max_verify_rounds():
                    break

                added = 0
                needs_search = bool(
                    verdict.get("needs_more_search")
                    or verdict.get("followup_query")
                    or verdict.get("missing_evidence")
                )
                if needs_search:
                    search_q = followup_query(q, verdict)
                    yield {"type": "status", "message": "Verification found gaps; searching again..."}
                    if local_on:
                        local_items, local_warnings = _gather_local_items(search_q, mode)
                        added += _extend_unique(items, local_items)
                        for w in local_warnings:
                            yield {"type": "warning", "message": w}
                    if is_web_search_enabled():
                        ext_items, ext_warnings = _gather_external_items(search_q, AGENTIC_EXTRA_SEARCH_K)
                        added += _extend_unique(items, ext_items)
                        for w in ext_warnings:
                            yield {"type": "warning", "message": w}
                    if added:
                        sources = _public_sources(items)
                        yield {"type": "sources", "sources": sources}
                    else:
                        yield {"type": "warning", "message": "Follow-up search did not find new sources."}
                else:
                    yield {"type": "status", "message": "Verification requested a rewrite; refining answer..."}

            final_answer = (answer or "(no answer)") + verification_footer(
                verdict=verdict,
                rounds=rounds_done,
                run_info=run_info,
            )
            answer_parts.append(final_answer)
            yield {"type": "token", "text": final_answer}
        else:
            yield {"type": "status", "message": "Writing the answer..."}
            evidence = format_evidence(items)
            user_msg = build_user_message(q, evidence)
            messages = history + [{"role": "user", "content": user_msg}]
            for chunk in provider.stream_chat(
                messages, system=SYSTEM_PROMPT, max_tokens=ANSWER_MAX_TOKENS, temperature=0.3
            ):
                answer_parts.append(chunk)
                yield {"type": "token", "text": chunk}
    except Exception as exc:
        msg = f"\n\n_Answer generation failed: {exc}_"
        answer_parts.append(msg)
        yield {"type": "token", "text": msg}

    answer = "".join(answer_parts).strip() or "(no answer)"
    sources = _public_sources(items)
    mem.append_turn(session_id, "assistant", answer, sources=sources)
    yield {"type": "done", "answer": answer}
