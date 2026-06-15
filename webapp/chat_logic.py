"""
Server-side chat orchestration for the web UI.

Reuses the existing backend (retrieval, LLM, memory) and yields a stream of
small JSON events that the browser renders. No backend code is modified here;
this module only wires the proven pieces together for the new UI.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

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

from backend.memory.store import MemoryStore, default_db_path, estimate_tokens  # noqa: E402
from backend.answering.query_sanity import check_query_sanity  # noqa: E402
from backend.answering.agentic_answer import (  # noqa: E402
    agentic_loop_enabled,
    auto_review_enabled,
    build_revision_message,
    complete_text,
    followup_query,
    max_verify_rounds,
    run_best_python_block,
    verification_passed,
    verify_answer,
)
from backend.llm.streaming_provider import get_provider  # noqa: E402
from backend.external_search import gather_external_evidence, is_web_search_enabled  # noqa: E402
from backend.observability import tracing  # noqa: E402  (no-op unless LANGFUSE_ENABLED=true)
from backend.answering.citations import repair_citations  # noqa: E402


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
        conv_env = os.getenv("CONVERSATIONS_DB_PATH")
        conv_path = (ROOT / conv_env) if conv_env else None
        _memory = MemoryStore(default_db_path(ROOT), conversations_path=conv_path)
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
    "- For code / implementation / simulation requests: you MAY use your OWN expert knowledge\n"
    "  of the algorithm to write COMPLETE, RUNNABLE, ORIGINAL code (imports + a small runnable\n"
    "  example). Do NOT refuse because the sources lack code. Cite sources for the surrounding\n"
    "  explanation; do not copy code verbatim from repositories; note any license constraints.\n"
    "- Prefer depth, accuracy, and breadth over brevity.\n"
    "- Write in clean, professional prose. Do NOT use emojis or decorative symbols.\n"
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


def _evidence_budget() -> int:
    """Total evidence chars allowed in the prompt — read live so the run mode applies."""
    return int(os.getenv("EVIDENCE_BUDGET_CHARS", "28000"))

_PROMPT_LIMIT_RE = re.compile(r"[Pp]rompt tokens limit exceeded:\s*(\d+)\s*>\s*(\d+)")


def _prompt_limit(message: str):
    """Parse a provider 'Prompt tokens limit exceeded: HAVE > ALLOWED' error."""
    m = _PROMPT_LIMIT_RE.search(message or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def format_evidence(sources: List[Dict[str, Any]], max_chars: int = EVIDENCE_CHARS_PER_SOURCE,
                    max_items: int = EVIDENCE_MAX_ITEMS, budget_chars: int | None = None) -> str:
    """Format local and/or external evidence items into a numbered, cited block,
    bounded to `max_items` sources and `budget_chars` total so the prompt stays
    affordable. Works on raw local retrieval dicts and external dicts."""
    if budget_chars is None:
        budget_chars = _evidence_budget()
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

AGENTIC_EXTRA_SEARCH_K = int(os.getenv("AGENTIC_EXTRA_SEARCH_K", "8"))


# How many external sources to keep, answer token budget, and deep-search planning.
# Read LIVE (not baked) so the run mode (fast/deep) applies per request.
def _external_top_k() -> int:
    return int(os.getenv("EXTERNAL_TOP_K", "20"))


def _answer_max_tokens() -> int:
    return int(os.getenv("ANSWER_MAX_TOKENS", "8000"))


def _deep_subqueries() -> int:
    """Number of extra "angle" sub-queries (0 in fast mode = just the literal query)."""
    return int(os.getenv("DEEP_SEARCH_SUBQUERIES", "3"))


def _deep_subquery_top_k() -> int:
    return int(os.getenv("DEEP_SUBQUERY_TOP_K", "6"))

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


def _deep_queries(question: str) -> List[str]:
    """The main question plus a few auto-planned sub-questions ('angles'), so every
    search is a mini deep-research sweep. Falls back to just the question."""
    if _deep_subqueries() <= 0:
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
    return [question] + extras[:_deep_subqueries()]


# ----------------------------------------------------------------------
# Compact conversation memory (Mem0-style layering): recent turns + a rolling
# summary of older turns + relevant facts, capped at a token budget. This shapes
# ONLY what is sent to the LLM — the full raw history is still saved for display +
# versioning. Falls back to recent-only on any failure; never breaks chat.
# ----------------------------------------------------------------------
def compact_memory_enabled() -> bool:
    return os.getenv("COMPACT_MEMORY", "true").strip().lower() not in ("0", "false", "no", "off")


def _int_env(name: str, default: int, lo: int) -> int:
    try:
        return max(lo, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def memory_recent_turns() -> int:
    return _int_env("MEMORY_RECENT_TURNS", 4, 0)


def memory_max_tokens() -> int:
    return _int_env("MEMORY_MAX_TOKENS", 3000, 200)


def memory_summary_stale() -> int:
    return _int_env("MEMORY_SUMMARY_STALE", 2, 1)


def memory_max_facts() -> int:
    return _int_env("MEMORY_MAX_FACTS", 6, 0)


_SUMMARY_SYSTEM = (
    "You maintain a COMPACT running summary of a chat so older turns can be dropped from the "
    "model's context without losing what matters. Fold the new older messages into the existing "
    "summary. Keep it under ~150 words: preserve decisions, named entities, the user's goal and "
    "preferences, and any open threads; drop pleasantries and redundant detail. Also extract 0-5 "
    "DURABLE facts about the user/project (stable across the chat), each a short key + value. "
    'Reply with ONLY strict JSON: {"summary": "...", "facts": [{"key": "...", "value": "..."}]}'
)


def _parse_summary_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _summarize_older(prev_summary: str, turns: List[Dict[str, str]]):
    """One timeout-bounded LLM call: fold `turns` into `prev_summary` and extract a few durable
    facts. Returns (summary, facts) or (None, []) on any unavailability/timeout/parse error, so
    the caller keeps the previous summary and chat never breaks."""
    from backend.llm.streaming_provider import get_provider
    try:
        provider = get_provider()
        if not provider.is_available:
            return None, []
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in turns if t.get("content"))[:6000]
        user = (f"EXISTING SUMMARY (may be empty):\n{prev_summary or '(none)'}\n\n"
                f"NEW OLDER MESSAGES to fold in:\n{convo}")

        def _run() -> str:
            parts: List[str] = []
            total = 0
            for tok in provider.stream_chat([{"role": "user", "content": user}],
                                            system=_SUMMARY_SYSTEM,
                                            max_tokens=_int_env("MEMORY_SUMMARY_MAX_TOKENS", 400, 64),
                                            temperature=0.2):
                if not isinstance(tok, str):
                    continue
                parts.append(tok)
                total += len(tok)
                if total > 4000:
                    break
            return "".join(parts)

        import concurrent.futures as cf
        timeout = float(os.getenv("MEMORY_SUMMARY_TIMEOUT", "8") or 8)
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            raw = ex.submit(_run).result(timeout=timeout)
    except Exception:                               # noqa: BLE001 - never break chat on summarize
        return None, []

    obj = _parse_summary_json(raw)
    if not obj:
        return None, []
    summary = (obj.get("summary") or "").strip()
    if not summary:
        return None, []
    facts = [f for f in (obj.get("facts") or []) if isinstance(f, dict)][:8]
    return summary, facts


def _hist_tokens(history: List[Dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in history)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    cap = max_tokens * 4
    return text if len(text) <= cap else text[: max(0, cap - 1)].rstrip() + "…"


def _format_memory_block(facts: List[Dict[str, Any]], summary: str) -> str:
    parts: List[str] = []
    if facts:
        lines = ["Relevant facts about the user / project:"]
        lines += [f"- {f['key']}: {f['value']}" for f in facts]
        parts.append("\n".join(lines))
    if summary:
        parts.append("Summary of earlier conversation (older turns, condensed):\n" + summary)
    if not parts:
        return ""
    return "\n\n[Conversation memory]\n" + "\n\n".join(parts)


def _build_compact_context(mem, session_id: str, question: str) -> Dict[str, Any]:
    """Assemble compact LLM context: recent N turns verbatim + a rolling summary of older turns
    (refreshed only when stale) + relevant facts, capped at MEMORY_MAX_TOKENS. Returns
    {system_extra, history, tokens, summary}. Shapes ONLY the LLM context; never raises."""
    recent_n = memory_recent_turns()
    try:
        all_turns = mem.get_turns(session_id)
    except Exception:
        all_turns = []
    # Drop the current question (and, on regenerate, its prior answer) from the tail — the
    # question is re-added by the caller as the evidence-augmented user message.
    trimmed = list(all_turns)
    if trimmed and trimmed[-1].get("role") == "assistant":
        trimmed = trimmed[:-1]
    if trimmed and trimmed[-1].get("role") == "user":
        trimmed = trimmed[:-1]
    recent = trimmed[len(trimmed) - recent_n:] if recent_n > 0 else []
    older = trimmed[: len(trimmed) - len(recent)]
    history = [{"role": t["role"], "content": t["content"]} for t in recent]

    if not compact_memory_enabled():
        return {"system_extra": "", "history": history, "tokens": _hist_tokens(history), "summary": ""}

    # SUMMARY — refresh only when enough new OLDER turns have accumulated (else reuse the cache).
    summ = mem.get_session_summary(session_id)
    summary, upto = summ["summary"], summ["upto"]
    unsummarized = older[upto:] if upto <= len(older) else older
    if older and len(unsummarized) >= memory_summary_stale():
        new_summary, facts = _summarize_older(summary, unsummarized)
        if new_summary is not None:                 # success -> persist; failure -> keep old summary
            summary = new_summary
            try:
                mem.set_session_summary(session_id, summary, len(older))
            except Exception:
                pass
            for f in facts:
                k = str(f.get("key") or "").strip()[:60]
                v = str(f.get("value") or "").strip()[:300]
                if k and v:
                    try:
                        mem.upsert_fact("session", k, v, session_id=session_id)
                    except Exception:
                        pass

    # FACTS — only the ones relevant to THIS question.
    try:
        facts_rel = mem.relevant_facts(session_id, question, limit=memory_max_facts())
    except Exception:
        facts_rel = []

    # TOKEN BUDGET — assembled (facts + summary + recent) <= MEMORY_MAX_TOKENS. Recent turns are
    # the freshest, so keep them; if recent alone is over, drop the oldest; then COMPRESS the older
    # summary to fit; only if still over (large facts) drop facts. Always keeps >= 1 recent turn.
    budget = memory_max_tokens()
    if _hist_tokens(history) > budget:
        while len(history) > 1 and _hist_tokens(history) > budget:
            history.pop(0)

    def _assemble(_facts, _summary):
        se = _format_memory_block(_facts, _summary)
        return se, estimate_tokens(se) + _hist_tokens(history)

    system_extra, total = _assemble(facts_rel, summary)
    while summary and total > budget:               # compress the older summary first
        over = total - budget
        summary = _truncate_to_tokens(summary, max(0, estimate_tokens(summary) - over - 1))
        system_extra, total = _assemble(facts_rel, summary)
    if total > budget and facts_rel:                # last resort: drop the (small) facts block
        facts_rel = []
        system_extra, total = _assemble(facts_rel, summary)
    logger.info("compact memory: recent=%d turn(s), older=%d, summary=%d tok, facts=%d, "
                "assembled=%d tok (budget=%d)", len(history), len(older),
                estimate_tokens(summary), len(facts_rel), total, budget)
    return {"system_extra": system_extra, "history": history, "tokens": total, "summary": summary}


# ----------------------------------------------------------------------
# Code-intent route: run the autonomous code agent and stream its events
# ----------------------------------------------------------------------
def _run_code_agent(question: str, session_id: str, mem,
                    q_version_id: Optional[int] = None,
                    node_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Stream a code-agent run (write -> run in sandbox -> verify against generated tests) for a
    code-intent query. The final code is saved as an ANSWER VERSION under the question version
    `q_version_id` (created here if not supplied, so the function also works standalone). Runs the
    agent off-thread so events stream live."""
    if q_version_id is None:
        _info = mem.start_question(session_id, question)
        q_version_id, node_id = _info["turn_id"], _info["node_id"]
    import queue
    import threading
    from backend.agent.loop import run_agent, result_to_markdown

    # Compact memory for the code agent too: rolling summary of older turns + recent turns.
    _ctx = _build_compact_context(mem, session_id, question)
    _convo_parts: List[str] = []
    if _ctx["summary"]:
        _convo_parts.append("Summary of earlier conversation:\n" + _ctx["summary"])
    _convo_parts += [f"{t['role']}: {t['content']}" for t in _ctx["history"] if t.get("content")]
    conversation = "\n".join(_convo_parts)[:3000]

    ev: "queue.Queue" = queue.Queue()
    DONE = object()
    box: Dict[str, Any] = {}

    def worker():
        try:
            box["res"] = run_agent(question, use_search=True,
                                   conversation=conversation, on_event=ev.put)
        except Exception as exc:
            ev.put({"type": "error", "message": str(exc)})
        finally:
            ev.put(DONE)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        e = ev.get()
        if e is DONE:
            break
        yield e

    res = box.get("res")
    content = result_to_markdown(res) if res is not None else "_(the code agent produced no result)_"
    av = None
    try:
        av = mem.add_answer_version(q_version_id, content)
    except Exception:
        pass
    done = {"type": "done", "answer": content, "code_agent": True}
    if av:
        done.update({"node_id": node_id, "qversion_id": q_version_id,
                     "answer_turn_id": av["turn_id"], "answer_version_index": av["version_index"],
                     "answer_total": av["total"]})
    yield done


# ----------------------------------------------------------------------
# The streaming orchestration
# ----------------------------------------------------------------------
def stream_chat_events(
    session_id: str,
    question: str,
    mode: str = "Default",
    top_k: int = 8,
    web_search: bool = True,
    *,
    edit_node_id: Optional[str] = None,
    regen_qversion_id: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield event dicts: sanity | status | sources | token | warning | version | done | error.

    Versioning (ChatGPT-style): `edit_node_id` adds a new question version under an existing
    node (edit / re-ask, keeping the old one); `regen_qversion_id` adds a new answer version
    under an existing question version (regenerate); neither set = a brand-new question node.

    Web search is the PRIMARY knowledge source. The local Oracle/PDF RAG is
    optional and off unless ENABLE_LOCAL_RAG=true, so the app runs in production
    with no Oracle database and no uploaded papers — just a web-search key + LLM.
    """
    q = (question or "").strip()
    mem = memory()
    user_id = mem.session_owner(session_id) or "local"

    # Regenerate reuses the existing question version's text (the client need not resend it).
    _regen_qv = None
    if regen_qversion_id is not None:
        _regen_qv = mem.get_version(regen_qversion_id)
        if not _regen_qv or _regen_qv.get("role") != "user":
            yield {"type": "error", "message": "That question version is no longer available."}
            return
        q = (_regen_qv.get("content") or "").strip()

    sanity = check_query_sanity(q)
    if not sanity.ok:
        yield {"type": "sanity", "message": sanity.user_message or "Please rephrase your question."}
        return

    # Apply the run profile (fast/deep) to the process env BEFORE planning/retrieval, so
    # the live knobs (sub-queries, external top-k, verify rounds, auto-review, budgets) take
    # effect for this request. Fast (default) is local-first + quick; deep does the full sweep.
    from backend.answering.research_modes import apply_research_mode, normalize_mode
    profile = apply_research_mode(mode)
    mode = normalize_mode(mode)
    logger.info("chat mode=%s (subqueries=%d, ext_top_k=%d, verify_rounds=%d, auto_review=%s)",
                profile["mode"], profile["deep_search_subqueries"], profile["external_top_k"],
                profile["agentic_max_verify_rounds"], profile["auto_review"])

    # Resolve the question version this answer belongs to (ChatGPT-style versioning). The stored
    # question keeps the user's EXACT text; only the search query below is refined.
    if regen_qversion_id is not None:                       # new answer under the same question
        q_version_id = int(regen_qversion_id)
        node_id = _regen_qv.get("node_id")
    elif edit_node_id:                                      # edit / re-ask -> new question version
        _info = mem.add_question_version(session_id, edit_node_id, q)
        q_version_id, node_id = _info["turn_id"], _info["node_id"]
        yield {"type": "version", "scope": "question", "node_id": node_id,
               "qversion_id": q_version_id, "version_index": _info["version_index"],
               "total": _info["total"]}
    else:                                                   # brand-new question node (version 1)
        _info = mem.start_question(session_id, q)
        q_version_id, node_id = _info["turn_id"], _info["node_id"]

    def _ans_meta(av: Dict[str, Any]) -> Dict[str, Any]:
        """Answer-version fields attached to the `done` event so the UI updates its switcher."""
        return {"node_id": node_id, "qversion_id": q_version_id,
                "answer_turn_id": av["turn_id"], "answer_version_index": av["version_index"],
                "answer_total": av["total"]}

    # Silently fix spelling/grammar BEFORE search so typos don't poison retrieval.
    # The question version stored above keeps the user's exact words; everything downstream
    # (intent, embedding, retrieval, external search, the LLM) uses the corrected text.
    from backend.answering.query_refine import refine_query
    q = refine_query(q)

    # Code-intent queries go straight to the autonomous code agent — never the prose/citation
    # pipeline (which would wrongly refuse for "sources lack code" and ship a toy demo). The agent
    # proves correctness by running tests. Routing is SEMANTIC (task_classifier): it recognizes any
    # request to write/run/simulate/benchmark/model/compute in any domain or phrasing, and unions
    # with the regex is_code_intent for high recall. It degrades to pure regex when the LLM is
    # unavailable (or CODE_INTENT_SEMANTIC=false) and never raises.
    from backend.answering.task_classifier import classify
    if classify(q).code_task:
        yield from _run_code_agent(q, session_id, mem, q_version_id, node_id)
        return

    # One trace per chat request (no-op unless LANGFUSE_ENABLED=true). Carries only
    # coarse settings — never the question text.
    trace = tracing.start_trace("chat_request", mode=mode, top_k=top_k,
                                web_search=bool(web_search))

    # Embed the question ONCE (if semantic reuse is on); reused for lookup AND save.
    query_emb, query_meta = (None, None)
    cache_on = answer_cache_enabled() and not _freshness_sensitive(q)
    cached = None
    with trace.span("cache_check", enabled=cache_on) as _sp:
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
            _sp.set(hit=bool(cached))
    if cached:
        sources = cached.get("sources") or []
        answer = cached.get("answer") or ""
        pct = int(float(cached.get("similarity", 0.0)) * 100)
        kind = cached.get("match_kind", "lexical")
        mem.record_answer_cache_hit(int(cached["id"]))
        _av = mem.add_answer_version(q_version_id, answer, sources=sources)
        trace.set(cached=True).end()
        yield {"type": "status", "message":
               f"Reusing a saved answer from memory ({pct}% {kind} match)."}
        yield {"type": "sources", "sources": sources}
        yield {"type": "token", "text": answer}
        yield {"type": "done", "answer": answer, "cached": True,
               "similarity": pct, "match_kind": kind, **_ans_meta(_av)}
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

    # Records its own span (count metadata) from inside the worker thread; the trace
    # handle is captured explicitly so nesting is correct despite the thread hop.
    def _traced(span_name, fn, *fn_args):
        with trace.span(span_name) as sp:
            result = fn(*fn_args)
            try:
                sp.set(count=len(result[0]))
            except Exception:
                pass
            return result

    seen_warnings: set = set()
    for idx, query in enumerate(queries):
        tag = "your question" if idx == 0 else f"angle {idx}: {query[:64]}"
        # Local RAG and external search are independent and both blocking — run them
        # concurrently so the stage takes max(local, external), not their sum.
        t_stage = time.time()
        futures: Dict[str, concurrent.futures.Future] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            if local_on:
                yield {"type": "status", "message": f"Searching your papers — {tag}..."}
                futures["local"] = ex.submit(_traced, "local_rag", _gather_local_items, query, mode)
            if is_web_search_enabled():
                yield {"type": "status", "message":
                       f"Searching the web, research papers, patents & GitHub — {tag}..."}
                k = _external_top_k() if idx == 0 else _deep_subquery_top_k()
                futures["external"] = ex.submit(_traced, "external_search", _gather_external_items, query, k)
            results: Dict[str, Any] = {}
            for name, fut in futures.items():
                try:
                    # Hard backstop on external search so a stalled channel can't block the
                    # chat — local retrieval still returns a partial answer. (The orchestrator
                    # already caps channels; this guards the gather+rerank tail too.) Local
                    # retrieval has no timeout: it must finish for there to be a local answer.
                    timeout = None
                    if name == "external":
                        timeout = float(os.getenv("EXTERNAL_GATHER_TIMEOUT", "30")) + 8.0
                    results[name] = fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.info("external search exceeded its timeout; partial local answer")
                    yield {"type": "warning", "message": "External search timed out — answering from local sources."}
                    results[name] = ([], [])
                except Exception as exc:
                    logger.info("%s retrieval failed: %s", name, type(exc).__name__)
                    results[name] = ([], [])
        # Merge deterministically (local first) so ordering/citations stay stable.
        for name in ("local", "external"):
            if name not in results:
                continue
            got_items, got_warnings = results[name]
            _extend_unique(items, got_items)
            for w in got_warnings:
                if w not in seen_warnings:
                    seen_warnings.add(w)
                    yield {"type": "warning", "message": w}
        logger.info("retrieval stage (%s): %d sources in %.1fs",
                    tag, len(items), time.time() - t_stage)

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
        _av = mem.add_answer_version(q_version_id, msg, sources=[])
        trace.set(n_sources=0).end()
        yield {"type": "done", "answer": msg, **_ans_meta(_av)}
        return

    with trace.span("source_selection") as _sp:
        sources = _public_sources(items)
        _sp.set(n_sources=len(sources))
    yield {"type": "sources", "sources": sources}

    # Compact memory: recent turns verbatim + a rolling summary of older turns + relevant facts,
    # capped at a token budget (Mem0-style). This is the ONLY conversation context sent to the LLM;
    # the full raw history stays saved for display + versioning. sys_prompt carries facts+summary.
    _ctx = _build_compact_context(mem, session_id, q)
    history = _ctx["history"]
    sys_prompt = SYSTEM_PROMPT + _ctx["system_extra"]

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
            for round_no in range(1, max_verify_rounds() + 1):

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
                with trace.span("prompt_build", round=round_no) as _sp:
                    budget = _evidence_budget()
                    evidence = format_evidence(items, budget_chars=budget)
                    _sp.set(evidence_chars=len(evidence), n_sources=len(items))
                answer = ""
                with trace.span("llm_stream", round=round_no) as _sp:
                    for _shrink in range(5):
                        err = None
                        try:
                            parts = []
                            for piece in provider.stream_chat(
                                    _messages_for(evidence), system=sys_prompt,
                                    max_tokens=_answer_max_tokens(), temperature=0.3, yield_reasoning=True):
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
                    _sp.set(model=getattr(provider, "model", None), output_len=len(answer),
                            tokens_out_est=len(answer) // 4)   # ~4 chars/token (no exact usage)

                yield {"type": "status", "message": "Checking for runnable Python simulation..."}
                with trace.span("code_simulation") as _sp:
                    run_info = run_best_python_block(answer)
                    if run_info:
                        _sp.set(attempted=bool(run_info.get("attempted")),
                                ok=bool(run_info.get("ok")),
                                summary=run_info.get("summary"))
                if run_info:
                    if run_info.get("attempted"):
                        yield {"type": "status", "message": f"Sandbox result: {run_info.get('summary')}"}
                    else:
                        yield {"type": "warning", "message": run_info.get("summary", "Simulation was not run.")}

                yield {"type": "status", "message": "Verifying answer against the retrieved evidence..."}
                try:
                    with trace.span("agentic_verify", round=round_no) as _sp:
                        verdict = verify_answer(
                            provider,
                            question=q,
                            evidence=evidence,
                            answer=answer,
                            run_info=run_info,
                        )
                        _sp.set(score=int(verdict.get("score", 0)), ok=bool(verdict.get("ok")))
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

            # Automatic peer review (the "Review" step, run for you): critique the final
            # answer (with topical relevance), improve it once if it's weak. Reviewer jargon
            # (novelty/soundness numbers, recommendation) is never shown to the user.
            review_offtopic = False
            if auto_review_enabled() and answer and answer.strip() and answer != "(no answer)":
                yield {"type": "status", "message": "Reviewing the answer…"}
                with trace.span("auto_review") as _sp:
                    try:
                        from backend.answering.reviewer import review as _peer_review, is_relevant
                        rev = _peer_review(answer, task=q)
                    except Exception:
                        rev = None
                    if rev and not rev.get("error"):
                        review_offtopic = not is_relevant(rev)
                        _sp.set(recommendation=rev.get("recommendation"), relevant=not review_offtopic)
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
                                    system=sys_prompt, max_tokens=_answer_max_tokens(), temperature=0.3)
                                if improved.strip():
                                    answer = improved
                                    answer_rewritten = True
                            except Exception:
                                pass

            clean_body = answer or ""
            if not (answer or "").strip():
                # Empty draft (e.g. an OpenRouter 402 capped output to ~0 tokens): show a
                # real, actionable message instead of "(no answer)" + a fake verification.
                final_answer = (
                    "The model returned an empty answer. This usually means the request "
                    "exceeded the provider's token budget — for example an OpenRouter 402 "
                    "\"can only afford N tokens\" on a low-credit account. Try a model that "
                    "has credits (a local Ollama model is free), or lower `ANSWER_MAX_TOKENS` "
                    "and `EVIDENCE_BUDGET_CHARS` in `.env`."
                )
            else:
                final_answer = answer   # clean body; no internal verifier/review jargon
            answer_parts.append(final_answer)
            yield {"type": "token", "text": final_answer}
            # Below the verification bar (or flagged off-topic) -> one clean styled warning,
            # never raw "(40/100, 5 round(s))" or "minor revision (novelty 7 ...)".
            if (answer or "").strip() and ((verdict and not verification_passed(verdict)) or review_offtopic):
                yield {"type": "low_confidence", "message": (
                    "This answer couldn't be fully verified against the available sources — "
                    "treat the key claims with caution and double-check anything critical.")}
        else:
            provider_ok = True
            yield {"type": "status", "message": "Writing the answer..."}
            with trace.span("prompt_build") as _sp:
                evidence = format_evidence(items)
                user_msg = build_user_message(q, evidence)
                _sp.set(evidence_chars=len(evidence), n_sources=len(items))
            messages = history + [{"role": "user", "content": user_msg}]
            with trace.span("llm_stream") as _sp:
                for chunk in provider.stream_chat(
                    messages, system=sys_prompt, max_tokens=_answer_max_tokens(),
                    temperature=0.3, yield_reasoning=True
                ):
                    if isinstance(chunk, dict):
                        yield {"type": "thinking", "text": chunk.get("reasoning", "")}
                    else:
                        answer_parts.append(chunk)
                        yield {"type": "token", "text": chunk}
                clean_body = "".join(answer_parts)
                _sp.set(model=getattr(provider, "model", None), output_len=len(clean_body),
                        tokens_out_est=len(clean_body) // 4)
    except Exception as exc:
        gen_failed = True
        msg = f"\n\n_Answer generation failed: {exc}_"
        answer_parts.append(msg)
        yield {"type": "token", "text": msg}

    answer = "".join(answer_parts).strip() or "(no answer)"
    with trace.span("memory_save") as _sp:
        sources = _public_sources(items)
        # Citation guard: strip any [n] that references a source outside the returned list,
        # so the saved/cached answer's citations always match the actual sources. (The
        # frontend strips out-of-range [n] from the live display too.)
        answer, removed_citations = repair_citations(answer, len(sources))
        _av = mem.add_answer_version(q_version_id, answer, sources=sources)

        # Save for reuse ONLY when the generation truly succeeded: provider worked, no
        # exception, the agentic answer passed verification AND its code didn't fail, and
        # the answer wasn't rewritten post-verification. Cache the clean body (no footers).
        verified = (not agentic_loop_enabled()) or (verification_passed(verdict) and not loop_run_failed)
        body = (clean_body or "").strip() or _strip_answer_footers(answer)
        body, _ = repair_citations(body, len(sources))
        did_cache = False
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
            did_cache = True
        _sp.set(cached=did_cache, citations_removed=len(removed_citations))
    trace.set(cached=did_cache, n_sources=len(sources)).end()
    if removed_citations:
        logger.info("citation guard: removed out-of-range %s (only %d sources)",
                    removed_citations, len(sources))
        yield {"type": "citation_warning", "removed": removed_citations, "n_sources": len(sources)}
    yield {"type": "done", "answer": answer, **_ans_meta(_av)}
