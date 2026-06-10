"""
Server-side chat orchestration for the web UI.

Reuses the existing backend (retrieval, LLM, memory) and yields a stream of
small JSON events that the browser renders. No backend code is modified here;
this module only wires the proven pieces together for the new UI.
"""
from __future__ import annotations

import os
import re
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
    auto_review_enabled,
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    "You are a meticulous, broad-domain research assistant. You answer questions on ANY\n"
    "topic by synthesizing the numbered source excerpts in the user's message.\n"
    "The sources come from EVERYWHERE and are tagged by type: (paper) research paper,\n"
    "(web) web page, (github) a repository file, (pdf) an online PDF, plus patents and\n"
    "encyclopedic entries. Treat ALL source types as equally valid evidence — do NOT\n"
    "favor any single type or any 'local' corpus; use the best evidence wherever it\n"
    "comes from, across the whole set.\n"
    "\n"
    "Write the best possible answer:\n"
    "- Lead with a direct, correct answer in 1-2 sentences, then expand into a thorough,\n"
    "  well-structured explanation — short sections and bullet points covering the key\n"
    "  facets (what it is, how it works, why it matters, trade-offs, alternatives, and\n"
    "  the current state of the art). Address the question from multiple angles.\n"
    "- SYNTHESIZE across sources: combine and compare what different sources say; call\n"
    "  out agreement, disagreement, and recency. For 'latest/current' questions, prefer\n"
    "  the most recent and authoritative sources and mention dates.\n"
    "- Cite every non-trivial claim with [1], [2], ... matching the numbered sources,\n"
    "  drawing on a DIVERSITY of sources rather than leaning on one.\n"
    "- Ground all specifics (equations, numbers, parameters, names, dates) in the cited\n"
    "  sources. Never invent facts, numbers, URLs, titles, or citations.\n"
    "- If the sources genuinely don't cover part of the question, say so plainly and\n"
    "  answer what you can from what is available.\n"
    "- For code / implementation / simulation requests: read the method from the cited\n"
    "  sources, then write COMPLETE, RUNNABLE, ORIGINAL code (imports + a small runnable\n"
    "  example) and explain each step, citing the source. Do NOT copy code verbatim from\n"
    "  repositories; reimplement the idea and note any license constraints.\n"
    "- Prefer depth, accuracy, and breadth over brevity.\n"
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
# Bound how much evidence is put in the prompt so deep search (many sources) stays
# affordable and fits the model's context: at most this many sources / total chars.
EVIDENCE_MAX_ITEMS = int(os.getenv("EVIDENCE_MAX_ITEMS", "16"))
EVIDENCE_BUDGET_CHARS = int(os.getenv("EVIDENCE_BUDGET_CHARS", "28000"))

_PROMPT_LIMIT_RE = re.compile(r"[Pp]rompt tokens limit exceeded:\s*(\d+)\s*>\s*(\d+)")


def _prompt_limit(message: str):
    """Parse a provider 'Prompt tokens limit exceeded: HAVE > ALLOWED' error."""
    m = _PROMPT_LIMIT_RE.search(message or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def format_evidence(sources: List[Dict[str, Any]], max_chars: int = EVIDENCE_CHARS_PER_SOURCE,
                    max_items: int = EVIDENCE_MAX_ITEMS, budget_chars: int = EVIDENCE_BUDGET_CHARS) -> str:
    """Format local and/or external evidence items into a numbered, cited block,
    bounded to `max_items` sources and `budget_chars` total so the prompt stays
    affordable. Works on raw local retrieval dicts and external dicts."""
    if not sources:
        return "(no retrieved sources)"
    parts: List[str] = []
    used = 0
    for i, r in enumerate(sources[:max_items], 1):
        text = (r.get("text") or r.get("chunk_text") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        block = _evidence_header(i, r) + "\n" + text
        if parts and used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block) + 2
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

# Deep research, always on: auto-decompose every question into a few "angles" and
# search each across all sources, so the answer is built from broad evidence — not
# just the literal query. Set DEEP_SEARCH_SUBQUERIES=0 to disable.
DEEP_SEARCH_SUBQUERIES = int(os.getenv("DEEP_SEARCH_SUBQUERIES", "3"))
DEEP_SUBQUERY_TOP_K = int(os.getenv("DEEP_SUBQUERY_TOP_K", "6"))

# Saved-answer reuse: exact/similar questions can be answered from SQLite memory
# without spending LLM or search tokens. Defaults are intentionally conservative.
ANSWER_CACHE_FRESHNESS_TERMS = (
    "latest", "current", "currently", "today", "tonight", "tomorrow",
    "yesterday", "now", "recent", "newest", "this week", "this month",
    "this year",
)


def answer_cache_enabled() -> bool:
    return _env_flag("ENABLE_ANSWER_CACHE", True)


def answer_cache_min_similarity() -> float:
    # High floor: lexical similarity alone is unreliable (a swap like "A vs B" can
    # score 0.95), so we require near-exact lexical OR a semantic match + the
    # unsafe_to_reuse guard in the store.
    try:
        return max(0.92, min(1.0, float(os.getenv("ANSWER_CACHE_MIN_SIMILARITY", "0.97"))))
    except ValueError:
        return 0.97


def answer_cache_semantic_enabled() -> bool:
    return _env_flag("ENABLE_ANSWER_CACHE_SEMANTIC", True)


def answer_cache_min_semantic() -> float:
    try:
        return max(0.80, min(1.0, float(os.getenv("ANSWER_CACHE_MIN_SEMANTIC", "0.88"))))
    except ValueError:
        return 0.88


def _query_embedding(question: str):
    """(vector, meta) for semantic cache matching, or (None, None) on any failure
    (missing GEMINI_API_KEY, missing deps, network error) — falls back to lexical."""
    if not answer_cache_semantic_enabled():
        return None, None
    try:
        from backend.common.embeddings import embed_query, provider as _emb_provider
        vec = embed_query(question)
        if not vec:
            return None, None
        meta = f"{_emb_provider()}:{os.getenv('EMBEDDING_MODEL', '')}:{len(vec)}"
        return vec, meta
    except Exception:
        return None, None


def answer_cache_max_age_seconds() -> float | None:
    try:
        days = float(os.getenv("ANSWER_CACHE_MAX_AGE_DAYS", "30"))
    except ValueError:
        days = 30.0
    if days <= 0:
        return None
    return days * 24 * 60 * 60


def answer_cache_limit() -> int:
    try:
        return max(20, min(1000, int(os.getenv("ANSWER_CACHE_CANDIDATE_LIMIT", "200"))))
    except ValueError:
        return 200


_FRESHNESS_RE = re.compile(
    r"\b(20\d{2}|latest|current|currently|today|tonight|tomorrow|yesterday|now|"
    r"recent|recently|newest|new(est)?|as of|up[- ]to[- ]date|state[- ]of[- ]the[- ]art|"
    r"this (week|month|year|quarter)|release[ds]?|version)\b"
)


def _freshness_sensitive(question: str) -> bool:
    """Time-sensitive questions bypass the cache so they always re-search.
    Errs toward bypassing (a missed cache is cheaper than a stale 'latest' answer)."""
    if _env_flag("ANSWER_CACHE_ALLOW_FRESHNESS_QUERIES", False):
        return False
    return bool(_FRESHNESS_RE.search((question or "").lower()))


def _strip_answer_footers(text: str) -> str:
    """Cache the answer BODY only — drop the appended auto-review / verification
    footers (they describe a live run and shouldn't be replayed as part of the answer)."""
    cut = len(text or "")
    for marker in ("\n\n**Auto-review:**", "\n\nVerification:"):
        i = (text or "").find(marker)
        if i != -1:
            cut = min(cut, i)
    return (text or "")[:cut].strip()


def _cacheable_answer(question: str, answer: str, sources: List[Dict[str, Any]]) -> bool:
    if not answer_cache_enabled() or _freshness_sensitive(question):
        return False
    text = (answer or "").strip()
    if len(text) < 80:
        return False
    low = text.lower()
    failure_markers = (
        "answer generation failed",
        "i couldn't find relevant information",
        "no knowledge source is enabled",
        "the language model isn't available",
    )
    if any(marker in low for marker in failure_markers):
        return False
    return bool(sources)


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


def _review_footer(rev: Dict[str, Any]) -> str:
    """A compact one-line verdict from the automatic peer review."""
    rec = rev.get("recommendation") or "reviewed"
    scores = rev.get("scores") or {}
    score_txt = " · ".join(f"{k} {v}" for k, v in scores.items()) if scores else ""
    return f"\n\n**Auto-review:** {rec}" + (f" ({score_txt})" if score_txt else "") + "."


def _deep_queries(question: str) -> List[str]:
    """The main question plus a few auto-planned sub-questions ('angles'), so every
    search is a mini deep-research sweep. Falls back to just the question."""
    if DEEP_SEARCH_SUBQUERIES <= 0:
        return [question]
    try:
        from backend.agent.research_agent import _plan
        provider = get_provider()
        if not provider.is_available:
            return [question]
        subs = _plan(provider, question)
    except Exception:
        return [question]
    ql = question.strip().lower()
    extras = [s for s in subs if s.strip() and s.strip().lower() != ql]
    return [question] + extras[:DEEP_SEARCH_SUBQUERIES]


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
    user_id = mem.session_owner(session_id) or "local"
    mem.append_turn(session_id, "user", q)

    # Embed the question ONCE (if semantic reuse is on); reused for lookup AND save.
    query_emb, query_meta = (None, None)
    cache_on = answer_cache_enabled() and not _freshness_sensitive(q)
    if cache_on:
        query_emb, query_meta = _query_embedding(q)
        cached = mem.find_cached_answer(
            user_id=user_id,
            question=q,
            min_similarity=answer_cache_min_similarity(),
            query_embedding=query_emb,
            query_meta=query_meta,
            min_semantic=answer_cache_min_semantic(),
            max_age_seconds=answer_cache_max_age_seconds(),
            limit=answer_cache_limit(),
        )
        if cached:
            sources = cached.get("sources") or []
            answer = cached.get("answer") or ""
            pct = int(float(cached.get("similarity", 0.0)) * 100)
            kind = cached.get("match_kind", "lexical")
            mem.record_answer_cache_hit(int(cached["id"]))
            mem.append_turn(session_id, "assistant", answer, sources=sources)
            yield {"type": "status", "message":
                   f"Reusing a saved answer from memory ({pct}% {kind} match)."}
            yield {"type": "sources", "sources": sources}
            yield {"type": "token", "text": answer}
            yield {"type": "done", "answer": answer, "cached": True,
                   "similarity": pct, "match_kind": kind}
            return

    items: List[Dict[str, Any]] = []
    local_on = local_rag_enabled()

    # --- Deep research, automatically: plan a few angles, then search the main
    #     question AND every angle across all sources, merging the evidence so the
    #     answer is built from everything found (local papers + web + papers +
    #     patents + GitHub). ---
    queries = _deep_queries(q)
    if len(queries) > 1:
        yield {"type": "status", "message":
               f"Planning the research — exploring {len(queries)} angles..."}

    seen_warnings: set = set()
    for idx, query in enumerate(queries):
        tag = "your question" if idx == 0 else f"angle {idx}: {query[:64]}"
        if local_on:
            yield {"type": "status", "message": f"Searching your papers — {tag}..."}
            local_items, local_warnings = _gather_local_items(query, mode)
            _extend_unique(items, local_items)
            for w in local_warnings:
                if w not in seen_warnings:
                    seen_warnings.add(w)
                    yield {"type": "warning", "message": w}
        if is_web_search_enabled():
            yield {"type": "status", "message":
                   f"Searching the web, research papers, patents & GitHub — {tag}..."}
            k = EXTERNAL_TOP_K if idx == 0 else DEEP_SUBQUERY_TOP_K
            ext_items, ext_warnings = _gather_external_items(query, k)
            _extend_unique(items, ext_items)
            for w in ext_warnings:
                if w not in seen_warnings:
                    seen_warnings.add(w)
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
    verdict: Dict[str, Any] = {}
    gen_failed = False
    provider_ok = False
    loop_run_failed = False     # generated Python failed in the sandbox
    answer_rewritten = False    # auto-review replaced the answer post-verification
    clean_body = ""             # the answer body to cache (no review/verify footers)
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
            provider_ok = True
            answer = ""
            run_info: Dict[str, Any] | None = None
            rounds_done = 0
            for round_no in range(1, max_verify_rounds() + 1):
                rounds_done = round_no

                def _messages_for(ev: str) -> List[Dict[str, str]]:
                    if answer and verdict:
                        um = build_revision_message(
                            question=q, evidence=ev, previous_answer=answer,
                            verdict=verdict, run_info=run_info)
                    else:
                        um = build_user_message(q, ev)
                    return history + [{"role": "user", "content": um}]

                yield {"type": "status", "message": (
                    f"Agent loop {round_no}/{max_verify_rounds()}: drafting a grounded answer..."
                )}

                # Draft, shrinking the evidence to fit if the model rejects the prompt
                # as too large (e.g. a low-balance account) so the answer still gets written.
                budget = EVIDENCE_BUDGET_CHARS
                evidence = format_evidence(items, budget_chars=budget)
                answer = ""
                for _shrink in range(5):
                    err = None
                    try:
                        parts = []
                        for piece in provider.stream_chat(
                                _messages_for(evidence), system=SYSTEM_PROMPT,
                                max_tokens=ANSWER_MAX_TOKENS, temperature=0.3, yield_reasoning=True):
                            if isinstance(piece, dict):
                                yield {"type": "thinking", "text": piece.get("reasoning", "")}
                            else:
                                parts.append(piece)
                        answer = "".join(parts).strip()
                    except Exception as exc:
                        err = exc
                    # Shrink the evidence and retry if the prompt was rejected as too
                    # large, or the reply came back empty (a tiny budget starved it).
                    too_big = err is not None and bool(
                        _prompt_limit(str(err)) or "402" in str(err) or "afford" in str(err).lower())
                    starved = err is None and len(answer.strip()) < 40
                    if (too_big or starved) and budget > 5000:
                        lim = _prompt_limit(str(err)) if err else None
                        budget = max(4000, int(budget * (lim[1] / lim[0] if lim else 0.55)))
                        yield {"type": "status",
                               "message": "Trimming evidence to fit the model's token budget..."}
                        evidence = format_evidence(items, budget_chars=budget)
                        continue
                    if err is not None:
                        raise err
                    break

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
                loop_run_failed = run_failed

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

            # Automatic peer review (the "Review" step, run for you): critique the
            # final answer, improve it once if it's weak, then show the verdict.
            review_note = ""
            if auto_review_enabled() and answer and answer.strip() and answer != "(no answer)":
                yield {"type": "status", "message": "Reviewing the answer…"}
                try:
                    from backend.answering.reviewer import review as _peer_review
                    rev = _peer_review(answer)
                except Exception:
                    rev = None
                if rev and not rev.get("error"):
                    if (rev.get("recommendation") or "").lower() in ("major revision", "reject"):
                        yield {"type": "status", "message": "Improving the answer after review…"}
                        fixes = "; ".join((rev.get("weaknesses") or []) + (rev.get("suggestions") or []))[:800]
                        rmsg = build_revision_message(
                            question=q, evidence=evidence, previous_answer=answer,
                            verdict={"feedback": fixes, "missing_evidence": [], "citation_issues": []},
                            run_info=run_info)
                        try:
                            improved = complete_text(
                                provider, history + [{"role": "user", "content": rmsg}],
                                system=SYSTEM_PROMPT, max_tokens=ANSWER_MAX_TOKENS, temperature=0.3)
                            if improved.strip():
                                answer = improved
                                answer_rewritten = True
                        except Exception:
                            pass
                    review_note = _review_footer(rev)

            clean_body = answer or ""
            final_answer = (answer or "(no answer)") + review_note + verification_footer(
                verdict=verdict,
                rounds=rounds_done,
                run_info=run_info,
            )
            answer_parts.append(final_answer)
            yield {"type": "token", "text": final_answer}
        else:
            provider_ok = True
            yield {"type": "status", "message": "Writing the answer..."}
            evidence = format_evidence(items)
            user_msg = build_user_message(q, evidence)
            messages = history + [{"role": "user", "content": user_msg}]
            for chunk in provider.stream_chat(
                messages, system=SYSTEM_PROMPT, max_tokens=ANSWER_MAX_TOKENS,
                temperature=0.3, yield_reasoning=True
            ):
                if isinstance(chunk, dict):
                    yield {"type": "thinking", "text": chunk.get("reasoning", "")}
                else:
                    answer_parts.append(chunk)
                    yield {"type": "token", "text": chunk}
            clean_body = "".join(answer_parts)
    except Exception as exc:
        gen_failed = True
        msg = f"\n\n_Answer generation failed: {exc}_"
        answer_parts.append(msg)
        yield {"type": "token", "text": msg}

    answer = "".join(answer_parts).strip() or "(no answer)"
    sources = _public_sources(items)
    mem.append_turn(session_id, "assistant", answer, sources=sources)

    # Save for reuse ONLY when the generation truly succeeded: provider worked, no
    # exception, the agentic answer passed verification AND its code didn't fail, and
    # the answer wasn't rewritten post-verification. Cache the clean body (no footers).
    verified = (not agentic_loop_enabled()) or (verification_passed(verdict) and not loop_run_failed)
    body = (clean_body or "").strip() or _strip_answer_footers(answer)
    if (cache_on and provider_ok and not gen_failed and verified
            and not answer_rewritten and _cacheable_answer(q, body, sources)):
        mem.cache_answer(
            user_id=user_id,
            session_id=session_id,
            question=q,
            answer=body,
            sources=sources,
            embedding=query_emb,
            embedding_meta=query_meta,
        )
    yield {"type": "done", "answer": answer}
