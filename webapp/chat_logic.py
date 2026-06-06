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

from backend.memory.store import MemoryStore, default_db_path
from backend.answering.query_sanity import check_query_sanity
from backend.llm.streaming_provider import get_provider
from backend.external_search import gather_external_evidence, is_web_search_enabled

# Local PDF RAG (Oracle + embeddings + reranker) is OPTIONAL and off by default.
# Web search is the primary source. `hybrid_retrieve` / `apply_research_mode` are
# imported lazily inside the chat flow only when this is enabled, so a web-only
# production deploy doesn't need Oracle or the heavy ML dependencies.
ENABLE_LOCAL_RAG = (os.getenv("ENABLE_LOCAL_RAG", "false") or "").strip().lower() in ("1", "true", "yes", "on")

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
    "- For code/algorithm questions: explain the algorithm, cite the source, then write\n"
    "  ORIGINAL implementation code in this project's style — do NOT copy source code\n"
    "  verbatim from repositories — and note any license constraints if relevant.\n"
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


def format_evidence(sources: List[Dict[str, Any]], max_chars: int = 900) -> str:
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

# If no local paper clears this relevance bar, the answer "isn't in the papers",
# so we automatically fall back to external search (web / arXiv / patents / GitHub).
LOCAL_FOUND_SCORE = float(os.getenv("LOCAL_FOUND_SCORE", "0.45"))


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
    }


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
    local_found = False

    # --- 1) Local PDF papers first (when enabled; needs Oracle + indexed papers) ---
    if ENABLE_LOCAL_RAG:
        yield {"type": "status", "message": "Searching your papers..."}
        try:
            # Imported lazily so a web-only deploy needs no Oracle / heavy ML deps.
            from backend.answering.research_modes import apply_research_mode
            from backend.retrieval.hybrid_retrieve import hybrid_retrieve
            try:
                apply_research_mode(mode)
            except Exception:
                pass
            local = select_sources(hybrid_retrieve(q, top_k=SOURCE_MAX + 6) or [])
            local_items = [_local_evidence_item(r) for r in local]
            items.extend(local_items)
            local_found = any(li["score"] >= LOCAL_FOUND_SCORE for li in local_items)
        except Exception as exc:
            yield {"type": "warning", "message": f"Local paper search is unavailable: {exc}"}

    # --- 2) AUTOMATIC fallback: if the papers don't answer it (or there are no
    #        papers), search the web, research papers (arXiv), patents & GitHub. ---
    if not local_found and is_web_search_enabled():
        yield {"type": "status", "message": (
            "Not found in your papers — searching the web, research papers, patents & GitHub..."
            if ENABLE_LOCAL_RAG else
            "Searching the web, research papers, patents & GitHub...")}
        try:
            ext_sources, ext_warnings = gather_external_evidence(q, max_results=8)
        except Exception as exc:
            ext_sources, ext_warnings = [], [f"External search failed: {exc}"]
        for es in ext_sources:
            d = es.to_public()
            d["text"] = (es.text or es.snippet or "").strip()   # full text for the LLM
            items.append(d)
        for w in ext_warnings:
            yield {"type": "warning", "message": w}

    # --- Nothing available at all -> explain instead of guessing ---
    if not items:
        if not ENABLE_LOCAL_RAG and not is_web_search_enabled():
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

    sources = []
    for i, it in enumerate(items, 1):
        pub = dict(it)
        pub["n"] = i
        pub["text"] = (it.get("text") or "")[:600]
        sources.append(pub)
    yield {"type": "sources", "sources": sources}
    yield {"type": "status", "message": "Writing the answer..."}

    evidence = format_evidence(items)
    user_msg = build_user_message(q, evidence)
    recent = mem.get_recent_turns(session_id, n_messages=6)
    # Replace the just-stored bare question with the evidence-augmented version.
    history = recent[:-1] if recent and recent[-1]["role"] == "user" else recent
    messages = history + [{"role": "user", "content": user_msg}]

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
        else:
            for chunk in provider.stream_chat(
                messages, system=SYSTEM_PROMPT, max_tokens=2048, temperature=0.3
            ):
                answer_parts.append(chunk)
                yield {"type": "token", "text": chunk}
    except Exception as exc:
        msg = f"\n\n_Answer generation failed: {exc}_"
        answer_parts.append(msg)
        yield {"type": "token", "text": msg}

    answer = "".join(answer_parts).strip() or "(no answer)"
    mem.append_turn(session_id, "assistant", answer, sources=sources)
    yield {"type": "done", "answer": answer}
