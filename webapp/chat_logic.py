"""
Server-side chat orchestration for the web UI.

Reuses the existing backend (retrieval, LLM, memory) and yields a stream of
small JSON events that the browser renders. No backend code is modified here;
this module only wires the proven pieces together for the new UI.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.memory.store import MemoryStore, default_db_path
from backend.answering.query_sanity import check_query_sanity
from backend.answering.research_modes import apply_research_mode
from backend.retrieval.hybrid_retrieve import hybrid_retrieve
from backend.llm.streaming_provider import get_provider

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
    "Answer using ONLY the numbered source excerpts provided in the user's message.\n"
    "- Be precise, technical, and clear; prefer short paragraphs and bullet points.\n"
    "- If the sources do not address the question, say so plainly instead of guessing.\n"
    "- Never invent facts, numbers, or paper titles.\n"
    "- Cite every non-trivial claim with [1], [2], ... matching the numbered sources.\n"
)


def format_evidence(sources: List[Dict[str, Any]], max_chars: int = 900) -> str:
    if not sources:
        return "(no retrieved sources)"
    parts = []
    for i, r in enumerate(sources, 1):
        title = r.get("title") or "Untitled"
        section = r.get("section") or r.get("section_name") or "?"
        ps = r.get("page_start") or "?"
        pe = r.get("page_end") or "?"
        text = (r.get("text") or r.get("chunk_text") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        parts.append(f"[{i}] {title} -- {section} (pages {ps}-{pe})\n{text}")
    return "\n\n".join(parts)


def build_user_message(question: str, evidence: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Retrieved evidence from the uploaded papers:\n\n{evidence}\n\n"
        f"Answer the question using only the evidence above. Cite sources with [n]."
    )


# Sections that rarely contain real answers — drop them from the evidence so the
# shown/used sources stay relevant (References lists, acknowledgements, etc.).
_LOW_VALUE_SECTIONS = ("reference", "bibliograph", "acknowledg", "author contribution",
                       "funding", "conflict of interest", "appendix")


def _keep_relevant(results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Drop low-value sections, then keep the top_k most relevant. Falls back to
    the original list if filtering would leave nothing."""
    def is_low_value(r: Dict[str, Any]) -> bool:
        section = (r.get("section") or r.get("section_name") or "").lower()
        return any(key in section for key in _LOW_VALUE_SECTIONS)

    kept = [r for r in results if not is_low_value(r)]
    return (kept or results)[:top_k]


def public_source(r: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Trim a retrieval result down to what the UI needs to render a source card."""
    return {
        "n": i,
        "title": r.get("title") or "Untitled",
        "section": r.get("section") or r.get("section_name") or "",
        "page_start": r.get("page_start"),
        "page_end": r.get("page_end"),
        "text": (r.get("text") or r.get("chunk_text") or "").strip()[:600],
        "score": round(float(r.get("rerank_score") or 0.0), 3),
    }


# ----------------------------------------------------------------------
# The streaming orchestration
# ----------------------------------------------------------------------
def stream_chat_events(
    session_id: str,
    question: str,
    mode: str = "Balanced",
    top_k: int = 8,
) -> Iterator[Dict[str, Any]]:
    """Yield event dicts: sanity | status | sources | token | done | error."""
    q = (question or "").strip()

    sanity = check_query_sanity(q)
    if not sanity.ok:
        yield {"type": "sanity", "message": sanity.user_message or "Please rephrase your question."}
        return

    mem = memory()
    mem.append_turn(session_id, "user", q)

    yield {"type": "status", "message": "Searching your papers..."}
    try:
        apply_research_mode(mode)
    except Exception:
        pass

    try:
        # Retrieve a few extra so we can drop low-value sections and still keep top_k.
        k = int(top_k)
        results = hybrid_retrieve(q, top_k=k + 4) or []
    except Exception as exc:
        yield {"type": "error", "message": f"Retrieval failed: {exc}"}
        return

    results = _keep_relevant(results, k)
    sources = [public_source(r, i) for i, r in enumerate(results, 1)]
    yield {"type": "sources", "sources": sources}
    yield {"type": "status", "message": "Writing the answer..."}

    evidence = format_evidence(results)
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
