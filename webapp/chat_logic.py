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
from datetime import datetime
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
    answer_logic_version,
    auto_review_enabled,
    build_revision_message,
    cache_revalidate_enabled,
    complete_text,
    consistency_check,
    consistency_check_enabled,
    followup_query,
    has_actionable_feedback,
    has_concrete_gap,
    independent_check,
    independent_verify_enabled,
    is_truly_verified,
    max_deep_loops,
    python_blocks_in_order,
    max_verify_rounds,
    REASONING_ANSWER_SYSTEM,
    reconcile_answer,
    run_best_python_block,
    verification_passed,
    verify_answer,
)
from backend.llm.streaming_provider import get_provider  # noqa: E402
from backend.external_search import gather_external_evidence, is_web_search_enabled  # noqa: E402
from backend.observability import tracing  # noqa: E402  (no-op unless LANGFUSE_ENABLED=true)
from backend.answering.arithmetic_check import verify_calculation  # noqa: E402
from backend.answering.code_intent import is_self_contained_calculation  # noqa: E402
from backend.answering.effort import assess_effort, effort_scaling_enabled, is_complex  # noqa: E402
from backend.answering import experience as _experience  # noqa: E402
from backend.answering import acquired_knowledge as _acquired  # noqa: E402
from backend.answering import tuning as _tuning  # noqa: E402
from backend.answering import citation_verifier as _citeverify  # noqa: E402
from backend.common import request_context as _rc  # noqa: E402
from backend.answering.source_router import (  # noqa: E402
    REASONING as SR_REASONING,
    WEB as SR_WEB,
    decide_source,
    source_router_enabled,
)
from backend.answering.citations import find_citations, repair_citations  # noqa: E402
from backend.answering.evidence_grader import (  # noqa: E402
    NONE,
    PARTIAL,
    STRONG,
    crag_enabled,
    extract_algorithm_spec,
    grade_evidence,
    paper_is_thin,
)
from backend.answering.relevance_gate import (  # noqa: E402
    relevance_gate_enabled,
    relevant_source_indices,
)


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
    "- ATTRIBUTION: only credit a result, product, model, or claim to a person or organisation\n"
    "  when a cited source EXPLICITLY ties them together. If a source is about a DIFFERENT entity\n"
    "  than the one asked about (e.g. a model from another lab/company), do NOT present it as the\n"
    "  asked-about entity's work — say the evidence doesn't cover that entity instead.\n"
    "- RELEVANCE: use ONLY sources that directly address THIS question. A citation must directly\n"
    "  support the specific claim it is attached to — if a numbered source does not support a claim,\n"
    "  do NOT cite it and do NOT bend the answer to fit it. If NONE of the sources address the\n"
    "  question, answer DIRECTLY from your own knowledge AS IF NO SOURCES WERE PROVIDED: add NO\n"
    "  citations, do NOT mention the sources or their relevance, and do NOT add a 'limitations of the\n"
    "  evidence' / 'according to the provided sources' note. Just give the clean, correct answer.\n"
    "- SELF-CONTAINED questions (a calculation, unit conversion, derivation, definition, or standard\n"
    "  textbook / general-knowledge reasoning question whose answer does NOT depend on these specific\n"
    "  sources): ANSWER DIRECTLY and CORRECTLY from your own knowledge and reasoning, showing the\n"
    "  working — do NOT reply that the sources don't cover it, and do NOT add citations or a sources /\n"
    "  evidence-limitations section. A correct reasoned answer is exactly what's wanted; you do not\n"
    "  need a source to do arithmetic or apply standard knowledge.\n"
    "- If the sources don't cover part of the question, answer THAT part from your own knowledge when\n"
    "  it is self-contained or standard (with no citation for it), and say the evidence is insufficient\n"
    "  ONLY for an EXTERNAL or empirical fact you genuinely cannot reason out (a specific dataset value,\n"
    "  a paper's result, a current event) — never refuse a whole question you can simply reason out.\n"
    "- For code / implementation / simulation requests: you MAY use your OWN expert knowledge\n"
    "  of the algorithm to write COMPLETE, RUNNABLE, ORIGINAL code (imports + a small runnable\n"
    "  example). Do NOT refuse because the sources lack code. Cite sources for the surrounding\n"
    "  explanation; do not copy code verbatim from repositories; note any license constraints.\n"
    "- Prefer depth, accuracy, and breadth over brevity.\n"
    "- Write in clean, professional prose. Do NOT use emojis or decorative symbols.\n"
)


def _today_note() -> str:
    """A live current-date anchor prepended to the system prompt so 'today'/'this year'/'latest'
    are interpreted against the REAL date, not the model's training-era default (which made a
    'latest this year' question answer with old years)."""
    return (
        "Today's date is " + datetime.now().strftime("%Y-%m-%d") + ". Interpret 'today', 'now', "
        "'this year', 'latest', and 'recent' relative to THIS date — never assume an earlier year "
        "from your training data. If the newest available evidence predates the user's timeframe, "
        "say so explicitly (e.g. 'the most recent source found is from <year>') instead of "
        "presenting older information as current.\n\n"
    )


def _freshness_note(question: str) -> str:
    """Extra instruction for time-sensitive questions: lead with the newest dated item and, when the
    freshest source is older than the current year, open with an explicit caveat instead of dressing
    up old news as 'this year'."""
    if not _freshness_sensitive(question):
        return ""
    yr = datetime.now().year
    return (
        f"\nThis is a TIME-SENSITIVE question about the current state as of {yr}. Sort the evidence "
        f"by date and LEAD with the most recent, dated development. Do NOT describe a development "
        f"from an earlier year as happening 'this year'. If the newest source you have predates "
        f"{yr}, OPEN with a clear caveat: 'The most recent information in these sources is from "
        f"<year>; there may be newer developments not captured here.'\n"
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
    """Total evidence chars allowed in the prompt — per-request (run mode) via the request context."""
    return _rc.request_int("EVIDENCE_BUDGET_CHARS", 28000)

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
    return _rc.request_int("EXTERNAL_TOP_K", 20)


def _answer_max_tokens() -> int:
    return _rc.request_int("ANSWER_MAX_TOKENS", 8000)


def _deep_subqueries() -> int:
    """Number of extra "angle" sub-queries (0 in fast mode = just the literal query)."""
    return _rc.request_int("DEEP_SEARCH_SUBQUERIES", 3)


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
    # unsafe_to_reuse guard in the store. Env default; the eval-gated tuner may override it (bounded).
    try:
        base = max(0.92, min(1.0, float(os.getenv("ANSWER_CACHE_MIN_SIMILARITY", "0.97"))))
    except ValueError:
        base = 0.97
    return _tuning.tuned("ANSWER_CACHE_MIN_SIMILARITY", base)


def answer_cache_semantic_enabled() -> bool:
    return _env_flag("ENABLE_ANSWER_CACHE_SEMANTIC", True)


def answer_cache_min_semantic() -> float:
    try:
        base = max(0.80, min(1.0, float(os.getenv("ANSWER_CACHE_MIN_SEMANTIC", "0.88"))))
    except ValueError:
        base = 0.88
    return _tuning.tuned("ANSWER_CACHE_MIN_SEMANTIC", base)


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
    r"\b(20\d{2}|latest|current|currently|today|tonight|tomorrow|yesterday|now|nowadays|"
    r"recent|recently|newest|new(est)?|as of|up[- ]to[- ]date|state[- ]of[- ]the[- ]art|"
    r"these days|at present|breaking|trending|this (week|month|year|quarter)|"
    r"release[ds]?|version)\b"
)


def _freshness_sensitive(question: str) -> bool:
    """True for a time-sensitive question (an explicit recency cue: latest / current / newest / now /
    this year / a recent date / state-of-the-art now / ...). These ALWAYS bypass the cache and re-answer
    fresh — a stale 'latest/current' answer is worse than a slightly slower one. There is NO opt-out: a
    recency question must never be served from a stored entry. (Semantic recency with no keyword is
    additionally caught by the source router's 'web' route, which also runs before the cache.)"""
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


_EVIDENCE_REFUSAL_RE = re.compile(
    r"\b(the\s+)?(provided\s+|retrieved\s+|available\s+|numbered\s+|given\s+)?"
    r"(evidence|sources?|excerpts?|context|documents?|passages?|references?)\b"
    r"[^.]{0,60}?\b(do(es)?\s+not|don'?t|doesn'?t|did\s+not|didn'?t|lack|fail(s|ed)?\s+to|cannot|"
    r"can'?t|no\b|without)[^.]{0,40}?\b(contain|cover|address|include|mention|discuss|provide|have|"
    r"specify|describe|support|answer|information|detail)", re.I)


def _is_evidence_refusal(answer: str) -> bool:
    """True when the answer is essentially 'the evidence/sources don't cover this' — a non-answer that a
    self-contained / reasoning-answerable question must NOT receive (the retrieved sources were just
    irrelevant). Matched on the OPENING SENTENCE only: a real answer LEADS with the answer (per the
    system prompt), so a brief later caveat inside a genuine answer is not mistaken for a refusal."""
    text = (answer or "").strip()
    if not text:
        return False
    first_sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:240].lower()
    return bool(_EVIDENCE_REFUSAL_RE.search(first_sentence))


def _cacheable_answer(question: str, answer: str, sources: List[Dict[str, Any]]) -> bool:
    """A real answer is cacheable on its QUALITY, not on whether it carries sources — so a high-quality
    REASONING answer (no sources) is stored too. The verified flag (passed separately to cache_answer)
    governs reuse; this only screens out non-answers (too short, provider/refusal failure markers)."""
    if not answer_cache_enabled() or _freshness_sensitive(question):
        return False
    text = (answer or "").strip()
    if len(text) < 80:
        return False
    low = text.lower()
    failure_markers = (
        "answer generation failed",
        "i couldn't find relevant information",
        "i couldn't find current information",
        "i couldn't produce a confident answer",
        "no knowledge source is enabled",
        "the language model isn't available",
    )
    if any(marker in low for marker in failure_markers):
        return False
    return True


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
        # Imported lazily so a web-only deploy needs no Oracle / heavy ML deps. The run profile is read
        # from the request context (bound at entry); this may run in a retrieval worker thread, which
        # inherits that context via ContextThreadPoolExecutor — no process-global env needed.
        from backend.retrieval.hybrid_retrieve import hybrid_retrieve
        local = select_sources(hybrid_retrieve(query, top_k=SOURCE_MAX + 6) or [])
        return [_local_evidence_item(r) for r in local], []
    except Exception as exc:
        return [], [f"Local paper search is unavailable: {exc}"]


def _external_item(es: Any) -> Dict[str, Any]:
    d = es.to_public()
    d["text"] = (getattr(es, "text", "") or getattr(es, "snippet", "") or "").strip()
    return d


def _gather_external_items(query: str, max_results: int) -> tuple[List[Dict[str, Any]], List[str]]:
    # Freshness-sensitive queries ('latest/current/today/...') ALWAYS re-search and are never
    # cached — the same policy the answer cache uses (_freshness_sensitive), so a stale 'latest'
    # answer can't be built from memoized evidence within the TTL.
    fresh = _freshness_sensitive(query)
    if not fresh:
        cached = _ext_cache_get(query, max_results)
        if cached is not None:                        # already fetched this (query, k) -> reuse
            return cached
    try:
        ext_sources, warnings = gather_external_evidence(query, max_results=max_results)
    except Exception as exc:
        return [], [f"External search failed: {exc}"]
    items = [_external_item(es) for es in ext_sources]
    if not fresh:
        _ext_cache_put(query, max_results, items, warnings)
    return items, warnings


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


def _source_relevance_enabled() -> bool:
    return _env_flag("SOURCE_RELEVANCE_DISPLAY", True)


_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _prose_only(text: str) -> str:
    """Strip fenced and inline code so citation parsing ignores code subscripts like `arr[2]`."""
    return _INLINE_CODE_RE.sub(" ", _CODE_FENCE_RE.sub(" ", text or ""))


def _cited_source_numbers(text: str) -> set:
    """The set of source numbers the answer cited, including GROUPED citations like [1, 3]
    (delegates to citations.find_citations so it matches the rest of the citation system). Code
    blocks are excluded so an array index `arr[2]` is never counted as a citation."""
    return find_citations(_prose_only(text or ""))


def _relevant_sources(answer: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the sources the answer CITED — those are the ones that justify it, so the panel
    never lists fetched-but-off-topic results (e.g. biology hits for a maths question). Each kept
    source keeps its original `n` (the frontend resolves [n] by source.n, so gaps are fine). Falls
    back to all sources when the answer cited none (never blanks the panel)."""
    if not _source_relevance_enabled():
        return sources
    cited = _cited_source_numbers(answer)
    if not cited:
        return sources
    kept = [s for s in sources if s.get("n") in cited]
    return kept or sources


def _apply_relevance_gate(items: List[Dict[str, Any]], question: str, trace):
    """PRE-DRAFT relevance gate: keep only retrieved sources that genuinely address the question,
    so an irrelevant (topically-similar) source can never ground or be cited in the answer. Returns
    the filtered items — possibly EMPTY, in which case the caller's no-items branch answers from
    reasoning with no citation. Generator: yields status events, returns the filtered list.

    Fail-open: if the gate is off / the provider is unavailable / the judge can't be parsed, the
    judge returns ALL indices, so nothing is dropped and behaviour is unchanged."""
    if not items or not relevance_gate_enabled():
        return items
    try:
        provider = get_provider()
    except Exception:
        return items
    if not getattr(provider, "is_available", False):
        return items
    with trace.span("source_relevance_gate") as _sp:
        keep = relevant_source_indices(provider, question=question, items=items)
        _sp.set(kept=len(keep), total=len(items))
    if len(keep) >= len(items):                 # all relevant (or fail-open) -> unchanged
        return items
    filtered = [it for i, it in enumerate(items, 1) if i in keep]
    dropped = len(items) - len(filtered)
    if filtered:
        yield {"type": "status",
               "message": f"Set aside {dropped} retrieved source(s) that didn't address your "
                          f"question; using only the relevant ones."}
    else:
        yield {"type": "status",
               "message": "The retrieved sources didn't address your question — answering from "
                          "reasoning instead."}
    return filtered


# UI badge text per CRAG grade — the frontend renders {"type":"grade",...} as a small chip on
# the answer so the user can see, at a glance, where the answer came from.
_GRADE_BADGE = {
    STRONG: "From your library",
    PARTIAL: "Library + web",
    NONE: "From the web",
}


def _grade_event(grade: str, web_on: bool = True) -> Dict[str, Any]:
    """Structured CRAG-grade event for the UI badge (distinct from the human-readable status)."""
    if grade == STRONG:
        msg = "Answered entirely from your PDF library."
    elif grade == PARTIAL:
        msg = ("Your PDFs partially covered this — combined with web results."
               if web_on else "Your PDFs partially covered this.")
    else:
        msg = ("Not found in your PDFs — answered from the web."
               if web_on else "Not found in your PDFs.")
    return {"type": "grade", "grade": grade, "label": _GRADE_BADGE.get(grade, ""), "message": msg}


# ----------------------------------------------------------------------
# CRAG latency options (both OFF by default — they don't change results, only timing):
#   * speculative external prefetch: start the web search concurrently with local retrieval so a
#     PARTIAL/NONE grade doesn't pay local-then-external sequentially. STRONG discards the prefetch
#     (trading some web-search spend for latency — hence opt-in).
#   * grade cache: reuse a recent (items, grade) for the same question in a session (helps
#     regenerate, which bypasses the answer cache). TTL- and size-bounded.
# ----------------------------------------------------------------------
def crag_speculative_external() -> bool:
    return _env_flag("CRAG_SPECULATIVE_EXTERNAL", False)


def crag_grade_cache_enabled() -> bool:
    return _env_flag("CRAG_GRADE_CACHE", False)


def _grade_cache_ttl() -> float:
    try:
        return float(os.getenv("CRAG_GRADE_CACHE_TTL", "120"))
    except (TypeError, ValueError):
        return 120.0


def _ext_arg_for(i: int, query: str) -> int:
    return _external_top_k() if i == 0 else _deep_subquery_top_k()


def _start_speculative_external(queries: List[str], trace):
    """Kick off the external gather in the background; returns a (executor, future) handle."""
    ex = _rc.ContextThreadPoolExecutor(max_workers=1)   # worker inherits this request's settings
    fut = ex.submit(_gather_pass, queries, _gather_external_items, _ext_arg_for,
                    trace=trace, span_name="external_search",
                    timeout=_rc.request_float("EXTERNAL_GATHER_TIMEOUT", 30.0) + 8.0)
    return (ex, fut)


def _resolve_speculative(spec):
    ex, fut = spec
    try:
        return fut.result()
    finally:
        ex.shutdown(wait=False)


def _drop_speculative(spec):
    """STRONG grade: we prefetched external but don't need it — don't block on it."""
    ex, fut = spec
    fut.cancel()
    ex.shutdown(wait=False)


_GRADE_CACHE: Dict[tuple, tuple] = {}   # (session_id, mode, normalized_q) -> (ts, items, grade)


def _grade_cache_key(session_id: str, mode: str, q: str) -> tuple:
    return (session_id, mode, " ".join((q or "").lower().split()))


def _grade_cache_get(session_id: str, mode: str, q: str):
    ent = _GRADE_CACHE.get(_grade_cache_key(session_id, mode, q))
    if not ent:
        return None
    ts, cached_items, grade = ent
    if time.time() - ts > _grade_cache_ttl():
        _GRADE_CACHE.pop(_grade_cache_key(session_id, mode, q), None)
        return None
    return [dict(it) for it in cached_items], grade


def _grade_cache_put(session_id: str, mode: str, q: str, items, grade: str) -> None:
    if len(_GRADE_CACHE) > 256:                      # crude bound; correctness comes from the TTL
        _GRADE_CACHE.clear()
    _GRADE_CACHE[_grade_cache_key(session_id, mode, q)] = (time.time(), [dict(it) for it in items], grade)


def _agent_parallelism() -> int:
    """Bounded worker cap for independent agent-level concurrency (search angles + follow-up
    gathers). Capped (never unbounded) to protect the GPU and free-tier rate limits."""
    try:
        return max(1, min(8, int(os.getenv("AGENT_PARALLELISM", "4"))))
    except ValueError:
        return 4


def _followup_confidence_floor() -> float:
    """Minimum router confidence required to DIVERT a message away from search (answer it as a
    follow-up). The owner's hard rule is that a genuinely new question must still search, so this is
    a code-level safety net on top of the LLM verdict: below it, we treat the message as research.
    The regex fallback scores 0.5 (< default 0.6), so an LLM-down request always searches."""
    try:
        return max(0.0, min(1.0, float(os.getenv("FOLLOWUP_CONFIDENCE", "0.6"))))
    except (TypeError, ValueError):
        return 0.6


# Deixis / back-references that a real follow-up uses to point at the conversation. Deliberately
# EXCLUDES content words like "output"/"result" ("the output impedance" is a new question). Used
# only to VETO a confident-but-wrong LLM follow-up verdict, never to create one.
_CONTEXT_DEIXIS_RE = re.compile(
    r"\b(it|its|it's|this|that|these|those|they|them|their|above|previous(ly)?|earlier|"
    r"aforementioned|former|prior|same)\b"
    r"|\b(the|that|this) (code|answer|reply|snippet|program|script|solution|function|"
    r"implementation|example)\b", re.I)
_ELLIPSIS_START_RE = re.compile(
    r"^\s*(and|also|but|so|then|plus|what about|how about|why|ok|okay)\b", re.I)


def _plausibly_references_context(question: str) -> bool:
    """Permissive check: could this message PLAUSIBLY be a follow-up to the conversation? True if it
    carries any deixis/pronoun/back-reference or an elliptical opener, or is very short. Used to
    VETO a confident-but-wrong LLM 'context'/'code_output' verdict on a clearly self-contained NEW
    question — so such a question still searches no matter how sure the model was."""
    s = (question or "").strip()
    if not s:
        return False
    if len(s.split()) <= 4:                          # "why?", "and the complexity?" -> elliptical
        return True
    return bool(_CONTEXT_DEIXIS_RE.search(s) or _ELLIPSIS_START_RE.search(s))


# Per-(query, k) memo so the verify->rewrite loop never re-fetches the SAME external search across
# rounds (and so similar angles within a request reuse). A short TTL keeps it fresh — external
# results don't change within a single multi-loop request; correctness comes from the TTL.
_EXT_CACHE: Dict[tuple, tuple] = {}   # (normalized_q, k) -> (ts, items, warnings)


def _ext_cache_ttl() -> float:
    try:
        return max(0.0, float(os.getenv("EXTERNAL_GATHER_CACHE_TTL", "120")))
    except ValueError:
        return 120.0


def _ext_cache_key(query: str, k: int) -> tuple:
    return (" ".join((query or "").lower().split()), int(k))


def _ext_cache_get(query: str, k: int):
    ent = _EXT_CACHE.get(_ext_cache_key(query, k))
    if not ent:
        return None
    ts, items, warnings = ent
    if time.time() - ts > _ext_cache_ttl():
        _EXT_CACHE.pop(_ext_cache_key(query, k), None)
        return None
    return [dict(it) for it in items], list(warnings)


def _ext_cache_put(query: str, k: int, items, warnings) -> None:
    if _ext_cache_ttl() <= 0:
        return
    if len(_EXT_CACHE) > 256:                         # crude bound; the TTL owns correctness
        _EXT_CACHE.clear()
    _EXT_CACHE[_ext_cache_key(query, k)] = (time.time(), [dict(it) for it in items], list(warnings))


def _traced_span(trace, span_name: str, fn, *fn_args):
    """Run `fn(*fn_args)` inside a trace span, recording the result count. The trace handle is
    passed explicitly so nesting stays correct across the worker-thread hop."""
    with trace.span(span_name) as sp:
        result = fn(*fn_args)
        try:
            sp.set(count=len(result[0]))
        except Exception:
            pass
        return result


def _gather_pass(queries: List[str], gather_fn, arg_for, *, trace, span_name: str,
                 timeout: Optional[float] = None) -> tuple[List[Dict[str, Any]], List[str], bool]:
    """Run `gather_fn(query, arg_for(idx, query))` for every query concurrently and merge the
    results uniquely in query order (deterministic, best angle first). Each gather returns
    (items, warnings). Returns (merged_items, warnings, timed_out)."""
    items: List[Dict[str, Any]] = []
    warnings: List[str] = []
    timed_out = False
    t0 = time.time()
    # All queries (the main question + its angles) are submitted up front, so they run
    # CONCURRENTLY in the pool (bounded by AGENT_PARALLELISM); collecting in submission order
    # only fixes the merge order, not execution — wall-time is the slowest query, not the sum.
    workers = max(1, min(_agent_parallelism(), len(queries)))
    with _rc.ContextThreadPoolExecutor(max_workers=workers) as ex:   # workers inherit request settings
        ordered = [ex.submit(_traced_span, trace, span_name, gather_fn, q, arg_for(i, q))
                   for i, q in enumerate(queries)]
        for fut in ordered:                     # iterate in submission order -> stable merge
            try:
                got_items, got_warnings = fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                timed_out = True
                continue
            except Exception as exc:
                logger.info("%s retrieval failed: %s", span_name, type(exc).__name__)
                continue
            _extend_unique(items, got_items)
            warnings.extend(got_warnings)
    logger.info("%s gather: %d sources from %d quer%s in %.1fs (%d workers)",
                span_name, len(items), len(queries), "y" if len(queries) == 1 else "ies",
                time.time() - t0, workers)
    return items, warnings, timed_out


def _legacy_sweep(queries: List[str], *, local_on: bool, mode: str, trace,
                  items: List[Dict[str, Any]], seen_warnings: set) -> Iterator[Dict[str, Any]]:
    """The original concurrent local+external sweep, used when CRAG is off (or local RAG is off).
    Local and external run together per query so the stage takes max(local, external), not their
    sum. Appends merged evidence to `items` in place and yields status/warning events."""
    for idx, query in enumerate(queries):
        tag = "your question" if idx == 0 else f"angle {idx}: {query[:64]}"
        t_stage = time.time()
        futures: Dict[str, concurrent.futures.Future] = {}
        with _rc.ContextThreadPoolExecutor(max_workers=2) as ex:   # workers inherit request settings
            if local_on:
                yield {"type": "status", "message": f"Searching your papers — {tag}..."}
                futures["local"] = ex.submit(_traced_span, trace, "local_rag",
                                             _gather_local_items, query, mode)
            if is_web_search_enabled():
                yield {"type": "status", "message":
                       f"Searching the web, research papers, patents & GitHub — {tag}..."}
                k = _external_top_k() if idx == 0 else _deep_subquery_top_k()
                futures["external"] = ex.submit(_traced_span, trace, "external_search",
                                                _gather_external_items, query, k)
            results: Dict[str, Any] = {}
            for name, fut in futures.items():
                try:
                    # Hard backstop on external search so a stalled channel can't block the
                    # chat — local retrieval still returns a partial answer. Local retrieval has
                    # no timeout: it must finish for there to be a local answer.
                    timeout = None
                    if name == "external":
                        timeout = _rc.request_float("EXTERNAL_GATHER_TIMEOUT", 30.0) + 8.0
                    results[name] = fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.info("external search exceeded its timeout; partial local answer")
                    yield {"type": "warning",
                           "message": "External search timed out — answering from local sources."}
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


def _deep_queries(question: str) -> List[str]:
    """The main question plus a few auto-planned sub-questions ('angles'), so a genuinely complex
    question becomes a mini deep-research sweep. Effort scales to the question: when effort scaling is
    on, a SIMPLE single-intent question is NOT decomposed at all (no planner LLM call, no per-angle
    searches) — only a genuinely complex/multi-part question is planned, up to the DEEP_SEARCH_SUBQUERIES
    cap. Falls back to just the question on any error / when planning is disabled."""
    cap = _deep_subqueries()
    if cap <= 0:
        return [question]
    if effort_scaling_enabled() and not is_complex(question):   # simple -> answer the literal query alone
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
    return [question] + extras[:cap]


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

        timeout = float(os.getenv("MEMORY_SUMMARY_TIMEOUT", "8") or 8)
        with _rc.ContextThreadPoolExecutor(max_workers=1) as ex:   # worker inherits request settings
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


def _build_compact_context(mem, session_id: str, question: str, *, user_id: Optional[str] = None,
                           query_embedding: Optional[List[float]] = None,
                           query_meta: Optional[str] = None) -> Dict[str, Any]:
    """Assemble compact LLM context: recent N turns verbatim + a rolling summary of older turns
    (refreshed only when stale) + relevant facts + LEARNED LESSONS, capped at MEMORY_MAX_TOKENS.
    Returns {system_extra, history, tokens, summary, lesson_ids}. Shapes ONLY the LLM context; never
    raises. Pass user_id (+ the query embedding) to recall experience lessons for THIS question."""
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

    # LEARNED LESSONS (experience memory): recall the most useful past lessons for THIS question and
    # inject them into the draft prompt. Fail-open, and kept OUTSIDE the token-budget squeeze below
    # (lessons are short + high-value). lesson_ids let the caller reinforce the ones that produce a
    # verified answer. No-op (no user_id) on the code-agent / non-answering call sites.
    lessons_block, lesson_ids = (
        _experience.recall(mem, user_id=user_id, question=question,
                           query_embedding=query_embedding, query_meta=query_meta)
        if user_id else ("", []))

    if not compact_memory_enabled():
        return {"system_extra": lessons_block, "history": history,
                "tokens": _hist_tokens(history) + estimate_tokens(lessons_block),
                "summary": "", "lesson_ids": lesson_ids}

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
    return {"system_extra": system_extra + lessons_block, "history": history,
            "tokens": total + estimate_tokens(lessons_block), "summary": summary,
            "lesson_ids": lesson_ids}


# ----------------------------------------------------------------------
# Code-intent route: run the autonomous code agent and stream its events
# ----------------------------------------------------------------------
def _run_code_agent(question: str, session_id: str, mem,
                    q_version_id: Optional[int] = None,
                    node_id: Optional[str] = None, *,
                    paper_spec: str = "", paper_citation: str = "",
                    supplement_github: bool = True) -> Iterator[Dict[str, Any]]:
    """Stream a code-agent run (write -> run in sandbox -> verify against generated tests) for a
    code-intent query. The final code is saved as an ANSWER VERSION under the question version
    `q_version_id` (created here if not supplied, so the function also works standalone). Runs the
    agent off-thread so events stream live.

    CRAG code-from-paper: when `paper_spec` is given (an algorithm description extracted from the
    user's PDFs), it is passed to the agent as the `brief` (the spec to implement), and the saved
    answer is annotated with the source paper(s). `supplement_github` controls whether the agent
    also fetches GitHub reference implementations — set False when the paper alone is a complete
    spec, True (default) when it is thin or no paper is involved."""
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
    _uid = (mem.session_owner(session_id) or "local")    # record/reuse agent runs per owner

    ev: "queue.Queue" = queue.Queue()
    DONE = object()
    box: Dict[str, Any] = {}

    def worker():
        try:
            box["res"] = run_agent(question, brief=paper_spec, use_search=supplement_github,
                                   conversation=conversation, on_event=ev.put,
                                   result_memory=mem, user_id=_uid)
        except Exception as exc:
            ev.put({"type": "error", "message": str(exc)})
        finally:
            ev.put(DONE)

    threading.Thread(target=_rc.run_in_current_context(worker), daemon=True).start()
    while True:
        e = ev.get()
        if e is DONE:
            break
        yield e

    res = box.get("res")
    content = result_to_markdown(res) if res is not None else "_(the code agent produced no result)_"
    if paper_citation:                       # cite the paper the algorithm was implemented from
        content = f"> **Implemented from your research library:** {paper_citation}\n\n" + content
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
# Conversation-aware follow-up handling (answer from the chat, not a fresh search)
# ----------------------------------------------------------------------
SYSTEM_PROMPT_FOLLOWUP = (
    "You are continuing an ongoing conversation. The user's latest message is a FOLLOW-UP that "
    "refers back to what was already discussed. Answer it using the conversation so far — no new "
    "sources were searched because none are needed for a follow-up. Be direct and accurate; if the "
    "answer is already present in earlier turns, use it. Do not invent citations or external sources."
)


def _conversation_history(mem, session_id: str) -> List[Dict[str, str]]:
    """Prior turns (role/content) for this session, EXCLUDING the just-added current question
    (and, on regenerate, its prior answer). Empty on the first message. Never raises."""
    try:
        all_turns = mem.get_turns(session_id)
    except Exception:
        return []
    trimmed = list(all_turns)
    if trimmed and trimmed[-1].get("role") == "assistant":   # regen: drop the current answer
        trimmed = trimmed[:-1]
    if trimmed and trimmed[-1].get("role") == "user":        # drop the current question
        trimmed = trimmed[:-1]
    return [{"role": t["role"], "content": t.get("content") or ""}
            for t in trimmed if (t.get("content") or "").strip()]


def _run_prior_code(code: str) -> str:
    """Run a code block from the conversation in the Docker sandbox and format the output as the
    answer. Never raises; degrades to a clear message if the sandbox is unavailable."""
    try:
        from backend.agent.code_runner import docker_available, run_python
    except Exception as exc:
        return f"I couldn't run the earlier code — the sandbox runner is unavailable ({exc})."
    if not docker_available():
        return ("Docker isn't running, so I can't execute the earlier code to get its live output. "
                "Start Docker and ask again (the output shown with the code above is from its "
                "previous run).")
    try:
        timeout = int(os.getenv("AGENTIC_SIMULATION_TIMEOUT", os.getenv("AGENT_RUN_TIMEOUT", "30")))
    except ValueError:
        timeout = 30
    res = run_python(code, timeout=timeout)
    out = (res.stdout or "").strip()
    if res.ok:
        return ("Here is the output of the code from earlier, re-run just now in the sandbox:\n\n"
                f"```text\n{out or '(the program produced no output)'}\n```")
    err = (res.stderr or res.error or "").strip()
    body = ("I re-ran the code from earlier and it raised an error:\n\n"
            f"```text\n{err or 'unknown error'}\n```")
    if out:
        body += f"\n\nPartial output before it failed:\n\n```text\n{out}\n```"
    return body


def _emit_answer(mem, q_version_id, node_id, answer: str, *, code_agent: bool = False):
    """Save an answer version and build the matching `done` event (shared by the follow-up paths)."""
    av = None
    try:
        av = mem.add_answer_version(q_version_id, answer, sources=[])
    except Exception as exc:
        logger.warning("follow-up answer not persisted (add_answer_version failed): %s", exc)
    done: Dict[str, Any] = {"type": "done", "answer": answer}
    if code_agent:
        done["code_agent"] = True
    if av:
        done.update({"node_id": node_id, "qversion_id": q_version_id,
                     "answer_turn_id": av["turn_id"], "answer_version_index": av["version_index"],
                     "answer_total": av["total"]})
    return done


def _answer_prior_code_output(question: str, conv_history: List[Dict[str, str]], mem,
                              q_version_id, node_id) -> Iterator[Dict[str, Any]]:
    """Re-run the most recent code block from the conversation and answer with its output. Yields
    events; the generator's return value is True if it handled the request, False if no runnable
    code was found (so the caller falls back to a plain conversational answer)."""
    code = ""
    for t in reversed(conv_history):
        if t.get("role") == "assistant":
            blocks = python_blocks_in_order(t.get("content") or "")
            if blocks:
                code = blocks[-1]          # the LAST python fence = the canonical final program
                break
    if not code:
        return False
    yield {"type": "sources", "sources": []}        # follow-up: no external sources to show
    yield {"type": "status", "message": "Re-running the code from our conversation in the sandbox..."}
    answer = _run_prior_code(code)
    yield {"type": "token", "text": answer}
    yield _emit_answer(mem, q_version_id, node_id, answer, code_agent=True)
    return True


def _answer_from_conversation(question: str, mem, session_id: str,
                              q_version_id, node_id) -> Iterator[Dict[str, Any]]:
    """Answer a follow-up from the conversation context alone — no external search, no source pool."""
    _ctx = _build_compact_context(mem, session_id, question)
    history = _ctx["history"]
    sys_prompt = _today_note() + SYSTEM_PROMPT_FOLLOWUP + _ctx["system_extra"]
    yield {"type": "sources", "sources": []}        # clear any stale source panel
    provider = get_provider()
    parts: List[str] = []
    if not provider.is_available:
        msg = "The language model isn't available right now, so I can't answer the follow-up."
        parts.append(msg)
        yield {"type": "token", "text": msg}
    else:
        messages = history + [{"role": "user", "content": question}]
        try:
            for chunk in provider.stream_chat(messages, system=sys_prompt,
                                              max_tokens=_answer_max_tokens(), temperature=0.3,
                                              yield_reasoning=True):
                if isinstance(chunk, dict):
                    yield {"type": "thinking", "text": chunk.get("reasoning", "")}
                else:
                    parts.append(chunk)
                    yield {"type": "token", "text": chunk}
        except Exception as exc:
            m = f"\n\n_Answer generation failed: {exc}_"
            parts.append(m)
            yield {"type": "token", "text": m}
    answer = "".join(parts).strip() or "(no answer)"
    yield _emit_answer(mem, q_version_id, node_id, answer)


# ----------------------------------------------------------------------
# The streaming orchestration
# ----------------------------------------------------------------------
def _av_meta(node_id, q_version_id, av: Dict[str, Any]) -> Dict[str, Any]:
    """Answer-version fields for the `done` event (module-level so helpers can build it too)."""
    return {"node_id": node_id, "qversion_id": q_version_id, "answer_turn_id": av["turn_id"],
            "answer_version_index": av["version_index"], "answer_total": av["total"]}


def _enforce_conclusion_matches_work(provider, question: str, answer: str):
    """CONCLUSION-MATCHES-WORK gate: confirm the answer's final stated result equals what its own
    derivation yields; on a contradiction, reconcile the answer so the stated result is taken directly
    from the work (single source of truth). Returns (answer, consistency_ok, corrected, derived):
      - consistency_ok=False -> a contradiction that could NOT be reconciled (caller withholds 'verified')
      - corrected=True       -> the answer was rewritten to remove the contradiction
    Fail-open: a disabled / empty / already-consistent answer returns (answer, True, False, "")."""
    if not consistency_check_enabled() or not (answer or "").strip() or answer == "(no answer)":
        return answer, True, False, ""
    check = consistency_check(provider, question=question, answer=answer)
    if check.get("consistent") is not False:           # consistent (or nothing to check) -> unchanged
        return answer, True, False, ""
    derived = (check.get("derived_result") or "").strip()
    fixed = reconcile_answer(provider, question=question, answer=answer, check=check)
    if fixed.strip() and fixed.strip() != (answer or "").strip():
        return fixed.strip(), True, True, derived       # reconciled -> now internally consistent
    return answer, False, False, derived                # contradiction stands -> not verified


def _reasoning_fallback(q: str, mem, session_id: str, q_version_id, node_id, user_id: str,
                        cache_on: bool, query_emb, query_meta, trace,
                        *, no_sources_enabled: bool = False,
                        regenerated: bool = False) -> Iterator[Dict[str, Any]]:
    """No usable retrieved evidence — answer a SOLVABLE question from the model's own reasoning instead
    of refusing. Judge the answer's quality ORIGIN-INDEPENDENTLY (basis='reasoning'); return it when it
    is good, append an honest note when it genuinely needs external facts, and cache it WITH its quality
    status (reusable only if verified)."""
    def _ans_meta(av: Dict[str, Any]) -> Dict[str, Any]:
        return _av_meta(node_id, q_version_id, av)
    provider = get_provider()
    if not provider.is_available:
        msg = "The language model isn't available right now, so I can't answer this."
        yield {"type": "token", "text": msg}
        _av = mem.add_answer_version(q_version_id, msg, sources=[])
        trace.set(n_sources=0).end()
        yield {"type": "done", "answer": msg, **_ans_meta(_av)}
        return

    yield {"type": "status", "message": "Reasoning it out..."}
    _ctx = _build_compact_context(mem, session_id, q, user_id=user_id,
                                  query_embedding=query_emb, query_meta=query_meta)
    sys_prompt = _today_note() + REASONING_ANSWER_SYSTEM + _ctx["system_extra"]
    parts: List[str] = []
    try:
        for piece in provider.stream_chat(
                _ctx["history"] + [{"role": "user", "content": q}], system=sys_prompt,
                max_tokens=_answer_max_tokens(), temperature=0.3, yield_reasoning=True):
            if isinstance(piece, dict):
                yield {"type": "thinking", "text": piece.get("reasoning", "")}
            else:
                parts.append(piece)
                yield {"type": "token", "text": piece}
    except Exception as exc:                                   # noqa: BLE001 - degrade, don't crash
        logger.info("reasoning fallback draft failed: %s", exc)
    answer = "".join(parts).strip()
    if not answer:
        # Some models stream the whole answer on the REASONING channel and nothing on the content
        # channel, leaving `parts` empty. Retry ONCE as a plain completion (no reasoning split) before
        # giving up — a question the model can answer must NOT be refused over a streaming quirk.
        try:
            answer = complete_text(
                provider, _ctx["history"] + [{"role": "user", "content": q}],
                system=sys_prompt, max_tokens=_answer_max_tokens(), temperature=0.3).strip()
            if answer:
                yield {"type": "token", "text": answer}
        except Exception:                                      # noqa: BLE001 - degrade, don't crash
            answer = ""
    if not answer:
        msg = "I couldn't produce a confident answer for this."
        yield {"type": "token", "text": msg}
        _av = mem.add_answer_version(q_version_id, msg, sources=[])
        trace.set(n_sources=0).end()
        yield {"type": "done", "answer": msg, **_ans_meta(_av)}
        return

    # SIMPLE, CORRECT, DIRECT: the answer is computed ONCE by the draft. Do NOT run an LLM 're-derivation'
    # or 'reconcile' pass here — those can hallucinate a different value and OVERRIDE correct work (the
    # "computes 1.44, then declares 14.4" bug). The only post-processing is the DETERMINISTIC arithmetic
    # SOURCE-OF-TRUTH: compute the answer's own shown work in code and override any stated number that
    # differs from the computed value (it recomputes, so it can never introduce a wrong value), then
    # propagate that one result to its restatements. A light quality check decides only whether to cache.
    _calc = verify_calculation(answer)
    answer = _calc.fixed_text

    yield {"type": "status", "message": "Checking the answer's quality..."}
    try:
        verdict = verify_answer(provider, question=q, evidence="", answer=answer, basis="reasoning")
    except Exception:                                          # noqa: BLE001 - never drop a real answer
        verdict = {"ok": True, "score": 100}
    # NOT VERIFIED IF MISMATCH: an answer whose stated result couldn't be reconciled to the computed
    # value (or that mixes unit conventions) is not cached as verified, even if the LLM judge liked it.
    good = verification_passed(verdict) and _calc.verified
    if not _calc.verified and _calc.notes:
        logger.info("arithmetic check withheld 'verified': %s", "; ".join(_calc.notes))
    if not good and verdict.get("needs_more_search"):
        note = "\n\n_Note: parts of this may need up-to-date or external information I don't have"
        if no_sources_enabled:
            note += " (enable web or local sources in `.env` for document-grounded answers)"
        note += "; treat the above as best-effort reasoning._"
        yield {"type": "token", "text": note}
        answer = (answer + note).strip()

    _av = mem.add_answer_version(q_version_id, answer, sources=[])
    if cache_on and _cacheable_answer(q, answer, []):
        try:
            mem.cache_answer(user_id=user_id, session_id=session_id, question=q,
                             answer=_strip_answer_footers(answer), sources=[],
                             embedding=query_emb, embedding_meta=query_meta, verified=good,
                             logic_version=answer_logic_version())
        except Exception:                                      # noqa: BLE001 - caching is best-effort
            pass
    # LEARN from this run: a MISTAKE lesson if the arithmetic source-of-truth corrected the answer, a
    # PREFERENCE lesson if this was a regeneration the user kept; reinforce the recalled lessons that
    # helped. Fail-open — learning never breaks the turn.
    try:
        _experience.capture_outcome(
            mem, user_id=user_id, question=q, answer=answer, corrections=_calc.corrections,
            regenerated=regenerated, verified=good, query_embedding=query_emb,
            query_meta=query_meta, logic_version=answer_logic_version())
        if good:
            mem.reinforce_lessons(_ctx.get("lesson_ids") or [])
    except Exception:                                          # noqa: BLE001 - learning never breaks a turn
        pass
    trace.set(n_sources=0).end()
    yield {"type": "done", "answer": answer, **_ans_meta(_av)}


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
        # Regenerate signals dissatisfaction with the prior answer: mark any cached copy NOT reusable so
        # it is never replayed; the fresh answer re-upgrades the record if it is judged high-quality.
        mem.downgrade_cached_answer(user_id, q)

    sanity = check_query_sanity(q)
    if not sanity.ok:
        yield {"type": "sanity", "message": sanity.user_message or "Please rephrase your question."}
        return

    # Bind the run profile (fast/deep) to THIS request's context BEFORE planning/retrieval, so the live
    # knobs (sub-queries, external top-k, verify rounds, auto-review, budgets) take effect for this
    # request ONLY — never via process-global env, so concurrent Fast/Deep requests can't clobber each
    # other. The server endpoint may have already bound a fuller context (incl. the selected model); if
    # so we don't overwrite it. A direct caller (tests / eval) binds here and the set persists across the
    # generator's yields in its own thread.
    from backend.answering.research_modes import resolve_research_mode, normalize_mode
    profile = resolve_research_mode(mode)
    if not _rc.has_request_settings():
        _rc.set_request_settings(profile)
    mode = normalize_mode(mode)
    logger.info("chat mode=%s (subqueries=%s, ext_top_k=%s, verify_rounds=%s, auto_review=%s)",
                profile["RESEARCH_MODE"], profile["DEEP_SEARCH_SUBQUERIES"], profile["EXTERNAL_TOP_K"],
                profile["AGENTIC_MAX_VERIFY_ROUNDS"], profile["AUTO_REVIEW"])

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

    # One trace per chat request (no-op unless LANGFUSE_ENABLED=true). Carries only
    # coarse settings — never the question text.
    trace = tracing.start_trace("chat_request", mode=mode, top_k=top_k,
                                web_search=bool(web_search))

    # Phase 3 — load any eval-PROVEN threshold overrides into the zero-latency cache (TTL-gated, so this
    # is a DB read at most once a minute, never per threshold). With no overrides set this is a no-op and
    # every getter returns its stock env/default. Fail-open.
    _tuning.refresh(mem)

    # Code-intent queries go straight to the autonomous code agent — never the prose/citation
    # pipeline (which would wrongly refuse for "sources lack code" and ship a toy demo). The agent
    # proves correctness by running tests. Routing is SEMANTIC (task_classifier): it recognizes any
    # request to write/run/simulate/benchmark/model/compute in any domain or phrasing, and unions
    # with the regex is_code_intent for high recall. It degrades to pure regex when the LLM is
    # unavailable (or CODE_INTENT_SEMANTIC=false) and never raises.
    from backend.answering.task_classifier import classify
    task_class = classify(q)            # reused below to gate the runnable-simulation check
    if task_class.code_task:
        # CRAG code-from-paper: if the requested algorithm lives in the user's PDFs, extract its
        # description and hand it to the code agent as the spec (cited in the answer). GitHub
        # references supplement only when the paper is thin; otherwise the paper alone is the spec.
        # When the algorithm is not in the PDFs, fall back to the GitHub-reference code path.
        paper_spec = paper_citation = ""
        supplement = True
        if crag_enabled() and local_rag_enabled():
            # Search the PDFs across the question AND its auto-planned angles (deep mode), so an
            # algorithm described across several sections/papers is assembled, not just whatever the
            # single literal query hit. Fast mode -> one query (no planner call), unchanged.
            code_queries = _deep_queries(q)
            yield {"type": "status", "message":
                   ("Checking your papers for the algorithm across "
                    f"{len(code_queries)} angles...") if len(code_queries) > 1
                   else "Checking your papers for the algorithm..."}
            code_local, _cw, _ct = _gather_pass(
                code_queries, _gather_local_items, lambda i, x: mode,
                trace=trace, span_name="local_rag")
            code_grade = grade_evidence(code_local)
            if code_grade in (STRONG, PARTIAL):
                paper_spec, paper_citation = extract_algorithm_spec(code_local, q)
            if paper_spec:
                supplement = (code_grade == PARTIAL) or paper_is_thin(code_local)
                note = f"Found the algorithm in your PDFs ({paper_citation or 'your library'}) — implementing and testing it"
                if supplement:
                    note += ", with GitHub references to fill gaps"
                yield {"type": "status", "message": note + "..."}
            else:
                yield {"type": "status",
                       "message": "Not in your PDFs — writing it with GitHub references..."}
        yield from _run_code_agent(q, session_id, mem, q_version_id, node_id,
                                   paper_spec=paper_spec, paper_citation=paper_citation,
                                   supplement_github=supplement)
        return

    # --- Conversation-aware routing: a follow-up that refers to the chat so far is answered FROM
    #     the conversation (or by re-running earlier code), NOT by a fresh web/paper sweep that
    #     pulls dozens of off-topic sources. Runs only when prior conversation exists, and defaults
    #     to "research" on any doubt so a genuinely new question still searches. (Code-intent
    #     queries already returned to the code agent above, which is itself conversation-aware.) ---
    search_q = q                       # the query used for retrieval; resolved if it's a follow-up
    conv_history = _conversation_history(mem, session_id)
    if conv_history:
        from backend.answering.conversation_router import route as route_conversation
        try:
            conv_route = route_conversation(q, conv_history)
        except Exception:
            conv_route = None
        if conv_route is not None:
            # Safety net for the owner's hard rule ("a new question must still search"). Divert to a
            # follow-up answer ONLY when ALL hold: the router is confident it is a follow-up; the
            # message PLAUSIBLY references the chat (a deixis veto, so a confident-but-wrong verdict
            # on a self-contained question can't skip search); and it is not time-sensitive (those
            # always need a fresh search — checked on the message AND its resolved form).
            resolved = (conv_route.resolved_query or "").strip()
            refs_context = _plausibly_references_context(q)
            # Use the anaphora-resolved query for retrieval ONLY when there is a reference to resolve;
            # for a standalone question keep the user's exact words (don't trust a stray LLM rewrite).
            search_q = resolved if (resolved and refs_context) else q
            fresh = _freshness_sensitive(q) or (bool(resolved) and _freshness_sensitive(resolved))
            divert = (conv_route.kind in ("code_output", "context")
                      and conv_route.confidence >= _followup_confidence_floor()
                      and refs_context
                      and not fresh)
            logger.info("conversation route: kind=%s conf=%.2f (%s) refs=%s divert=%s resolved=%r",
                        conv_route.kind, conv_route.confidence, conv_route.source, refs_context,
                        divert, resolved[:80])
            if divert and conv_route.kind == "code_output":
                handled = yield from _answer_prior_code_output(q, conv_history, mem,
                                                               q_version_id, node_id)
                if handled:
                    return
                yield from _answer_from_conversation(q, mem, session_id, q_version_id, node_id)
                return
            if divert:                 # context follow-up
                yield from _answer_from_conversation(q, mem, session_id, q_version_id, node_id)
                return

    # --- SOURCE ROUTER (runs BEFORE the cache, on purpose): decide WHICH source the question NEEDS. A
    #     TIME-SENSITIVE question (route 'web' — an explicit recency cue OR the LLM judging it needs
    #     current info) must NEVER be served from a stored answer (stale by definition), so the router
    #     runs AHEAD of the cache lookup and a 'web' route bypasses the cache entirely. Deterministic
    #     fast-paths (calculation -> reasoning, recency -> web) settle the obvious cases with no LLM
    #     call; the LLM router decides the rest and FAILS OPEN to 'corpus'. The verdict is LRU-cached,
    #     so a repeat question pays only an in-memory lookup. ---
    is_fresh = _freshness_sensitive(q)
    calc = is_self_contained_calculation(q)
    route_provider = get_provider() if (source_router_enabled() and not is_fresh and not calc) else None
    source = decide_source(route_provider, q, freshness=is_fresh, calc=calc)
    force_web = (source == SR_WEB)

    # --- EFFORT GAUGE (deterministic, no LLM): scale the research budget to what the question NEEDS.
    #     A simple, single-intent question gets 0 planned angles (see _deep_queries) and a single verify
    #     pass (no rewrite re-search); only a genuinely complex/multi-part question gets the full
    #     angle/loop budget. The fast/deep knobs stay the ceiling — the gauge only decides 0-vs-cap, so
    #     heavy work is the exception, not the default. effort.max_loops caps the agentic loop below. ---
    effort = assess_effort(
        q,
        angle_cap=_deep_subqueries(),
        loop_cap=min(max_verify_rounds(), max_deep_loops()),
    )
    logger.info("effort: %s (angles<=%d, loops<=%d, scaling=%s) for %r",
                effort.label, effort.angles, effort.max_loops, effort_scaling_enabled(), q[:80])

    # Embed the question ONCE (if semantic reuse is on); reused for lookup AND save. A time-sensitive
    # ('web') question is NEVER cached or served from cache — stale by definition; only stable
    # reasoning/corpus answers reuse.
    query_emb, query_meta = (None, None)
    cache_on = answer_cache_enabled() and not force_web
    reuse_on = cache_on and regen_qversion_id is None      # regenerate re-answers fresh; never replays
    cached = None
    # Embed when the cache OR the grown corpus (Phase 2) needs it — otherwise learned-passage recall
    # would silently degrade to lexical-only whenever the answer cache happens to be off.
    if cache_on or _acquired.acquired_enabled():
        query_emb, query_meta = _query_embedding(q)        # reuse lookup AND saving AND learned recall
    with trace.span("cache_check", enabled=reuse_on) as _sp:
        if reuse_on:
            cached = mem.find_cached_answer(
                user_id=user_id,
                question=q,
                min_similarity=answer_cache_min_similarity(),
                query_embedding=query_emb,
                query_meta=query_meta,
                min_semantic=answer_cache_min_semantic(),
                max_age_seconds=answer_cache_max_age_seconds(),
                limit=answer_cache_limit(),
                # Entries produced by OLDER answering logic are excluded here -> they re-answer below
                # so a deploy of answering fixes takes effect instead of replaying a stale answer.
                min_logic_version=answer_logic_version(),
            )
            _sp.set(hit=bool(cached))
    if cached:
        # NEVER serve a cached answer BLINDLY. A stored answer is a speed optimization for VERIFIED
        # answers only — it must still pass a lightweight conclusion-matches-work re-check before
        # reuse. If the stored answer now contradicts its own work, retire it and re-answer fresh
        # (the full pipeline below produces + verifies + re-caches a corrected answer).
        cached_answer = cached.get("answer") or ""
        serve_cached = True
        if cache_revalidate_enabled() and consistency_check_enabled() and cached_answer.strip():
            yield {"type": "status", "message": "Re-checking the saved answer before reusing it..."}
            provider = get_provider()
            if getattr(provider, "is_available", False):
                chk = consistency_check(provider, question=q, answer=cached_answer)
                if chk.get("consistent") is False:
                    serve_cached = False
                    mem.mark_cache_unverified(int(cached["id"]))    # retire it; don't serve again
                    logger.info("cached answer failed serve-time consistency re-check -> re-answering")
        if serve_cached:
            sources = cached.get("sources") or []
            answer = cached_answer
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
        # else: fall through to the full fresh pipeline (retrieval / reasoning + verification).

    # The source route was decided BEFORE the cache (above). 'reasoning' -> answer directly from the
    # model's own knowledge (no retrieval, no corpus citations, no "the sources" framing). A stable
    # reasoning answer was already served from cache above if one existed; otherwise produce it fresh.
    if source == SR_REASONING:
        no_src = not local_rag_enabled() and not is_web_search_enabled()
        yield {"type": "status", "message": "Answering directly from reasoning..."}
        yield from _reasoning_fallback(q, mem, session_id, q_version_id, node_id, user_id, cache_on,
                                       query_emb, query_meta, trace, no_sources_enabled=no_src,
                                       regenerated=regen_qversion_id is not None)
        return

    items: List[Dict[str, Any]] = []
    local_on = local_rag_enabled()
    crag_grade = NONE                 # set by the CRAG branch; read later by the Self-RAG escalation

    # --- WEB route (a recency cue, or the router judged the question needs current info): go WEB-ONLY,
    #     anchored to the present, so 'latest/current/this-year' is answered from fresh web sources and
    #     the static corpus / stale training content is never presented as current. (force_web already
    #     subsumes the recency cue, since a fresh question routes to 'web'.) ---
    if force_web and local_on and is_web_search_enabled():
        local_on = False
        logger.info("source=web -> web-only (skipping the static local library)")
    if force_web:
        _yr = str(datetime.now().year)
        if _yr not in search_q:
            search_q = f"{search_q} {_yr}"

    # --- Deep research, automatically: plan a few angles, then search the main
    #     question AND every angle across all sources, merging the evidence so the
    #     answer is built from everything found (local papers + web + papers +
    #     patents + GitHub). ---
    queries = _deep_queries(search_q)   # self-gates: 0 angles for a simple question (effort scaling on)
    if len(queries) > 1:
        yield {"type": "status", "message":
               f"Planning the research — exploring {len(queries)} angles..."}
        # List the angles so the user sees WHAT is being explored (they run concurrently, so
        # per-angle "now searching X" progress isn't meaningful — naming them up front is).
        for i, sub in enumerate(queries[1:], 1):
            yield {"type": "status", "message": f"  • angle {i}: {sub[:80]}"}

    seen_warnings: set = set()

    if crag_enabled() and local_on:
        # --- Corrective RAG: retrieve LOCAL papers first, GRADE the evidence, then ACT on the
        #     grade. STRONG answers from the PDFs alone (no external spend); PARTIAL keeps the PDF
        #     evidence and supplements with external search; NONE drops local and goes fully
        #     external. Grading reuses the reranker scores already on each chunk (no extra LLM). ---
        yield {"type": "status", "message": "Searching your papers..."}
        web_on = is_web_search_enabled()
        cached = _grade_cache_get(session_id, mode, q) if crag_grade_cache_enabled() else None

        # Optionally start the web search in parallel with local retrieval (opt-in; STRONG drops it).
        spec = None
        if cached is None and crag_speculative_external() and web_on:
            spec = _start_speculative_external(queries, trace)

        if cached is not None:
            local_items, crag_grade = cached
        else:
            local_items, local_warnings, _ = _gather_pass(
                queries, _gather_local_items, lambda i, q: mode,
                trace=trace, span_name="local_rag")
            for w in local_warnings:
                if w not in seen_warnings:
                    seen_warnings.add(w)
                    yield {"type": "warning", "message": w}
            crag_grade = grade_evidence(local_items)
            if crag_grade_cache_enabled():
                _grade_cache_put(session_id, mode, q, local_items, crag_grade)

        trace.set(crag_grade=crag_grade)
        logger.info("CRAG grade=%s (%d local chunks)", crag_grade, len(local_items))
        yield _grade_event(crag_grade, web_on)            # UI badge: where the answer comes from

        if crag_grade == STRONG:
            items.extend(local_items)
            yield {"type": "status", "message":
                   "Found a strong match in your PDFs — answering from your library."}
        elif crag_grade == PARTIAL:
            items.extend(local_items)
            yield {"type": "status", "message":
                   ("Your PDFs partially covered this, so I'm also searching the web, "
                    "research papers, patents & GitHub to fill the gaps...") if web_on else
                   "Your PDFs partially covered this (web search is off — answering from your library)."}
        else:  # NONE — drop the local evidence and go fully external
            yield {"type": "status", "message":
                   ("Not in your PDFs — searching the web, research papers, patents & GitHub..."
                    if web_on else
                    "I couldn't find this in your PDFs, and web search is off.")}

        if crag_grade in (PARTIAL, NONE) and web_on:
            if spec is not None:                          # use the prefetch started above
                ext_items, ext_warnings, timed_out = _resolve_speculative(spec)
                spec = None
            else:
                ext_items, ext_warnings, timed_out = _gather_pass(
                    queries, _gather_external_items, _ext_arg_for,
                    trace=trace, span_name="external_search",
                    timeout=_rc.request_float("EXTERNAL_GATHER_TIMEOUT", 30.0) + 8.0)
            if timed_out:
                yield {"type": "warning",
                       "message": "External search timed out — answering from available sources."}
            _extend_unique(items, ext_items)
            for w in ext_warnings:
                if w not in seen_warnings:
                    seen_warnings.add(w)
                    yield {"type": "warning", "message": w}
        if spec is not None:                              # STRONG (or web off): we don't need it
            _drop_speculative(spec)
    else:
        # CRAG off (or local RAG off): original concurrent sweep — local and external run
        # together per query so the stage takes max(local, external), not their sum.
        yield from _legacy_sweep(queries, local_on=local_on, mode=mode, trace=trace,
                                 items=items, seen_warnings=seen_warnings)

    # --- GROWN CORPUS (Phase 2): merge passages we LEARNED from earlier VERIFIED answers and that are
    #     relevant to THIS question — so we answer from a corpus that grew out of our own verified
    #     research, even when local RAG is off (no Oracle needed). Reuses the query embedding already
    #     computed (no extra call). SKIPPED for freshness-sensitive questions so a 'latest/current'
    #     query never gets served possibly-stale learned passages — those go live. Fail-open. ---
    if _acquired.acquired_enabled() and not _freshness_sensitive(q) and not force_web:
        acq_items = _acquired.recall_items(mem, user_id=user_id, question=q,
                                           query_embedding=query_emb, query_meta=query_meta)
        if acq_items:
            added = _extend_unique(items, acq_items)
            if added:
                yield {"type": "status",
                       "message": f"Recalled {added} learned source(s) from earlier verified answers."}

    # --- RELEVANCE GATE: keep only the retrieved sources that genuinely address the question.
    #     Topically-similar-but-irrelevant hits (which reranker scores can't catch) must never
    #     ground or be cited; if the gate empties `items`, the no-items branch below answers from
    #     reasoning with no spurious citation. ---
    items = yield from _apply_relevance_gate(items, q, trace)

    # --- Nothing retrieved (or nothing relevant) -> ANSWER FROM REASONING, don't refuse a
    #     solvable question; don't bend it to an irrelevant source ---
    if not items:
        no_src = not local_on and not is_web_search_enabled()
        yield {"type": "sources", "sources": []}
        if force_web or _freshness_sensitive(q):
            # Genuinely depends on current/external data we couldn't find -> honest, never fabricate
            # (and never fall back to stale corpus/training content presented as current).
            msg = "I couldn't find current information for that in the available sources."
            yield {"type": "token", "text": msg}
            _av = mem.add_answer_version(q_version_id, msg, sources=[])
            trace.set(n_sources=0).end()
            yield {"type": "done", "answer": msg, **_ans_meta(_av)}
            return
        # Self-contained / reasoning-answerable: answer from reasoning, judge it, return or refine.
        yield from _reasoning_fallback(q, mem, session_id, q_version_id, node_id, user_id, cache_on,
                                       query_emb, query_meta, trace, no_sources_enabled=no_src,
                                       regenerated=regen_qversion_id is not None)
        return

    with trace.span("source_selection") as _sp:
        sources = _public_sources(items)
        _sp.set(n_sources=len(sources))
    yield {"type": "sources", "sources": sources}

    # Compact memory: recent turns verbatim + a rolling summary of older turns + relevant facts,
    # capped at a token budget (Mem0-style). This is the ONLY conversation context sent to the LLM;
    # the full raw history stays saved for display + versioning. sys_prompt carries facts+summary.
    _ctx = _build_compact_context(mem, session_id, q, user_id=user_id,
                                  query_embedding=query_emb, query_meta=query_meta)
    history = _ctx["history"]
    sys_prompt = _today_note() + SYSTEM_PROMPT + _freshness_note(q) + _ctx["system_extra"]

    answer_parts: List[str] = []
    verdict: Dict[str, Any] = {}
    gen_failed = False
    provider_ok = False
    loop_run_failed = False     # generated Python failed in the sandbox
    answer_rewritten = False    # auto-review replaced the answer post-verification
    self_rag_escalated = False  # STRONG answer failed verification -> escalated to web once
    reroute_to_reasoning = False  # evidence draft refused a self-contained Q -> answer from reasoning
    consistency_ok = True       # stated result matched the answer's own derivation (or was reconciled)
    clean_body = ""             # the answer body to cache (no review/verify footers)
    learn_corrections = ()      # arithmetic overrides applied -> a mistake-lesson
    learn_reconciled = False    # conclusion-matches-work reconciled the answer -> a mistake-lesson
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
            # Cap the SEQUENTIAL verify->rewrite loop. By default DEEP_MAX_LOOPS matches
            # max_verify_rounds (fast=1, deep=3), so the cap does NOT reduce thoroughness; the
            # latency win comes entirely from the early-stop below (skip a rewrite when the draft
            # passes or the verifier names no concrete gap — an empty round can't improve the
            # answer). A query that genuinely needs every round still gets them. DEEP_MAX_LOOPS is
            # only a deliberate, operator-set speed/quality lever when set below max_verify_rounds.
            # The EFFORT gauge tightens this further (when enabled): a SIMPLE question runs a single
            # verify pass (effort.max_loops == 1), so it never re-searches — line ~1970 breaks at
            # round_no >= loop_cap before any follow-up/Self-RAG search. Only a complex question spends
            # the loop. Disabled (EFFORT_SCALING=off) -> the legacy mode cap applies to every question.
            loop_cap = min(max_verify_rounds(), max_deep_loops())
            if effort_scaling_enabled():
                loop_cap = min(loop_cap, effort.max_loops)
            loop_t0 = time.time()
            feedback_rewrites = 0       # guided rewrites done for feedback-only (no-gap) verdicts
            for round_no in range(1, loop_cap + 1):
                round_t0 = time.time()

                def _messages_for(ev: str) -> List[Dict[str, str]]:
                    if answer and verdict:
                        um = build_revision_message(
                            question=q, evidence=ev, previous_answer=answer,
                            verdict=verdict, run_info=run_info)
                    else:
                        um = build_user_message(q, ev)
                    return history + [{"role": "user", "content": um}]

                yield {"type": "status", "message": (
                    f"Agent loop {round_no}/{loop_cap}: drafting a grounded answer..."
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

                # GENERAL-KNOWLEDGE HANDOFF: this evidence-path draft cited NO source -> the model
                # answered from its OWN knowledge; the retrieved corpus wasn't genuinely used. Stop here
                # (before the evidence verify can fail it for lack of grounding, or rewrite it to force
                # citations / an evidence-limitations frame) and hand off to the clean reasoning path
                # post-loop. EXCEPTIONS keep their own machinery: a STRONG corpus match (the corpus is
                # genuinely relevant -> Self-RAG corroboration below), freshness/web, and code tasks.
                if (round_no == 1 and answer.strip() and not force_web and crag_grade != STRONG
                        and not _freshness_sensitive(q) and not task_class.code_task
                        and not _cited_source_numbers(answer)):
                    break

                # Runnable-Python check ONLY for code-intent queries. A pure research/reasoning
                # question (router said NON-code) skips this entirely — every loop — so we never
                # spin up the sandbox path for prose. Code tasks run their code in the dedicated
                # agent (they route there before this loop), so this stays correct for them too.
                if task_class.code_task:
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

                logger.info("agent loop %d/%d: round %.1fs (verify score %s, ok %s, run_failed %s)",
                            round_no, loop_cap, time.time() - round_t0,
                            verdict.get("score"), verdict.get("ok"), run_failed)

                if (verification_passed(verdict) and not run_failed) or round_no >= loop_cap:
                    break

                # Self-RAG: a STRONG answer drawn from the PDFs ALONE that fails verification means
                # the library wasn't enough after all. Escalate to external search ONCE and
                # regenerate with the merged evidence (the grade badge flips to "Library + web").
                # PARTIAL/NONE already searched externally, so the generic follow-up below covers them.
                if (crag_grade == STRONG and not self_rag_escalated
                        and not verification_passed(verdict)
                        and round_no < loop_cap and is_web_search_enabled()):
                    self_rag_escalated = True
                    yield {"type": "status", "message":
                           "Your PDFs didn't fully hold up — searching the web to corroborate "
                           "and revise the answer..."}
                    esc_items, esc_warnings, _esc_timed = _gather_pass(
                        queries, _gather_external_items,
                        lambda i, qq: (_external_top_k() if i == 0 else _deep_subquery_top_k()),
                        trace=trace, span_name="external_search",
                        timeout=_rc.request_float("EXTERNAL_GATHER_TIMEOUT", 30.0) + 8.0)
                    for w in esc_warnings:
                        if w not in seen_warnings:
                            seen_warnings.add(w)
                            yield {"type": "warning", "message": w}
                    if _extend_unique(items, esc_items):
                        yield {"type": "sources", "sources": _public_sources(items)}
                        yield _grade_event(PARTIAL, True)   # badge: needed the web after all
                    continue                                 # regenerate with the merged evidence

                # Early-stop: the draft didn't fully pass and the verifier named NO concrete,
                # structured gap (no missing evidence / citation issue / follow-up search). But the
                # verifier may still have given SPECIFIC prose feedback (e.g. "soften the overstated
                # claim about [2]") that a rewrite can fix from the existing evidence — so we do ONE
                # feedback-guided rewrite before finalizing. A second no-gap round would only chase a
                # vague target (the waste this removes). A code-run failure always continues to fix
                # the code. (FAST mode is loop_cap=1, so this never triggers a rewrite there.)
                if not loop_run_failed and not has_concrete_gap(verdict):
                    if has_actionable_feedback(verdict) and feedback_rewrites < 1:
                        feedback_rewrites += 1
                        yield {"type": "status",
                               "message": "Verification noted a specific fix; revising once from the evidence..."}
                        continue                         # rewrite next round using verdict feedback
                    yield {"type": "status", "message": "No concrete gap left to fix — finalizing the best answer."}
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
                    # Local + external follow-up run CONCURRENTLY (bounded pool), merged local-first
                    # so citation order stays stable. The external memo also skips a re-fetch if this
                    # same query was already searched earlier in the request.
                    fu_t0 = time.time()
                    fu_futs: Dict[str, concurrent.futures.Future] = {}
                    with _rc.ContextThreadPoolExecutor(max_workers=max(1, min(2, _agent_parallelism()))) as fx:
                        if local_on:
                            fu_futs["local"] = fx.submit(_gather_local_items, search_q, mode)
                        if is_web_search_enabled():
                            fu_futs["external"] = fx.submit(_gather_external_items, search_q, AGENTIC_EXTRA_SEARCH_K)
                        fu_res: Dict[str, Any] = {}
                        for _name, _fut in fu_futs.items():
                            try:
                                _to = None if _name == "local" else _rc.request_float("EXTERNAL_GATHER_TIMEOUT", 30.0) + 8.0
                                fu_res[_name] = _fut.result(timeout=_to)
                            except Exception as _exc:
                                logger.info("follow-up %s search failed: %s", _name, type(_exc).__name__)
                                # Surface it (don't swallow): the answer is built on fewer sources.
                                _msg = ("External follow-up search timed out — using available sources."
                                        if isinstance(_exc, concurrent.futures.TimeoutError)
                                        else f"Follow-up {_name} search failed.")
                                fu_res[_name] = ([], [_msg])
                    for _name in ("local", "external"):      # local first -> stable citation order
                        if _name not in fu_res:
                            continue
                        _gi, _gw = fu_res[_name]
                        added += _extend_unique(items, _gi)
                        for w in _gw:
                            yield {"type": "warning", "message": w}
                    logger.info("follow-up search: +%d sources in %.1fs", added, time.time() - fu_t0)
                    if added:
                        sources = _public_sources(items)
                        yield {"type": "sources", "sources": sources}
                    else:
                        yield {"type": "warning", "message": "Follow-up search did not find new sources."}
                else:
                    yield {"type": "status", "message": "Verification requested a rewrite; refining answer..."}

            logger.info("agentic answer: %d loop(s) of <=%d in %.1fs", round_no, loop_cap,
                        time.time() - loop_t0)

            # GENERAL-KNOWLEDGE HANDOFF (see the loop break above): an evidence-path answer that cites
            # NO source was produced from the model's OWN knowledge, not the corpus — route it to the
            # clean reasoning path below (no citations, no evidence-framing, no irrelevant sources,
            # origin-independent confidence) instead of auto-reviewing / emitting it as grounded.
            if ((answer or "").strip() and not force_web and crag_grade != STRONG
                    and not _freshness_sensitive(q) and not task_class.code_task
                    and not _cited_source_numbers(answer)):
                reroute_to_reasoning = True

            # Automatic peer review (the "Review" step, run for you): critique the final
            # answer (with topical relevance), improve it once if it's weak. Reviewer jargon
            # (novelty/soundness numbers, recommendation) is never shown to the user.
            review_offtopic = False
            if (auto_review_enabled() and not reroute_to_reasoning
                    and answer and answer.strip() and answer != "(no answer)"):
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
            # ROOT FIX: a self-contained / reasoning-answerable question whose evidence draft either
            # REFUSES ("the sources don't cover this") or simply CITES NOTHING (handoff flag above) was
            # not actually grounded in the corpus — do NOT ship it framed as evidence-based. Skip
            # emitting it and re-answer cleanly from REASONING below (no citations, no evidence-framing).
            if reroute_to_reasoning or (
                    (answer or "").strip() and _is_evidence_refusal(answer) and not _freshness_sensitive(q)):
                reroute_to_reasoning = True
            elif not (answer or "").strip():
                # Empty draft (e.g. an OpenRouter 402 capped output to ~0 tokens): show a
                # real, actionable message instead of "(no answer)" + a fake verification.
                final_answer = (
                    "The model returned an empty answer. This usually means the request "
                    "exceeded the provider's token budget — for example an OpenRouter 402 "
                    "\"can only afford N tokens\" on a low-credit account. Try a model that "
                    "has credits (a local Ollama model is free), or lower `ANSWER_MAX_TOKENS` "
                    "and `EVIDENCE_BUDGET_CHARS` in `.env`."
                )
                answer_parts.append(final_answer)
                yield {"type": "token", "text": final_answer}
            else:
                # ARITHMETIC SOURCE OF TRUTH (deterministic, FIRST, no LLM): compute the answer's own
                # shown work in code and OVERRIDE any 'EXPR = NUM' whose stated number differs from the
                # code-computed value of its expression. Runs on EVERY answer on this live path, so a
                # model-asserted equality result the system hasn't verified is never delivered. Safe by
                # construction (it only turns a shown equality true). A mixed unit convention (1024 vs
                # 1000) withholds 'verified' below; free-prose conclusions go to the LLM backstop next.
                _calc = verify_calculation(answer)
                if _calc.fixed_text != answer:
                    answer = _calc.fixed_text
                    clean_body = answer
                calc_ok = _calc.verified
                learn_corrections = _calc.corrections             # -> mistake-lesson if it overrode a value
                # CONCLUSION-MATCHES-WORK: an LLM backstop for NON-arithmetic conclusions (qualitative
                # claims the code check can't evaluate). Runs on the already code-corrected text, so it
                # can't override a number the deterministic check just made authoritative.
                if consistency_check_enabled():
                    yield {"type": "status", "message": "Checking the conclusion matches the work..."}
                    answer, consistency_ok, _corr, _dv = _enforce_conclusion_matches_work(provider, q, answer)
                    clean_body = answer
                    learn_reconciled = bool(_corr)                # -> mistake-lesson if it reconciled
                consistency_ok = consistency_ok and calc_ok       # arithmetic must also be consistent
                final_answer = answer   # clean body; no internal verifier/review jargon
                answer_parts.append(final_answer)
                yield {"type": "token", "text": final_answer}
                # Below the verification bar (or flagged off-topic) -> one clean styled warning,
                # never raw "(40/100, 5 round(s))" or "minor revision (novelty 7 ...)".
                if (verdict and not verification_passed(verdict)) or review_offtopic:
                    yield {"type": "low_confidence", "message": (
                        "This answer couldn't be fully verified against the available sources — "
                        "treat the key claims with caution and double-check anything critical.")}
                if not consistency_ok:
                    yield {"type": "low_confidence", "message": (
                        "The stated result appears to contradict the answer's own derivation and "
                        "couldn't be reconciled — treat the key result with caution.")}
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

    # The evidence draft refused a self-contained question (irrelevant sources) -> answer it from
    # REASONING instead of shipping the refusal. _reasoning_fallback yields its own answer + done.
    if reroute_to_reasoning:
        # The retrieved sources weren't genuinely used -> CLEAR the (irrelevant) source panel emitted
        # earlier, so the clean reasoning answer carries no spurious citations or sources.
        yield {"type": "sources", "sources": []}
        yield {"type": "status", "message": "Answering from general knowledge…"}
        yield from _reasoning_fallback(q, mem, session_id, q_version_id, node_id, user_id,
                                       cache_on, query_emb, query_meta, trace,
                                       regenerated=regen_qversion_id is not None)
        return

    answer = "".join(answer_parts).strip() or "(no answer)"
    with trace.span("memory_save") as _sp:
        sources = _public_sources(items)
        full_n = len(sources)
        # Citation guard: strip any [n] that references a source outside the FULL retrieved list,
        # so the saved/cached answer's citations always match the actual sources. (The frontend
        # strips out-of-range [n] from the live display too.) Done against full_n BEFORE the
        # relevance filter, since cited numbers index the full list.
        answer, removed_citations = repair_citations(answer, full_n)
        # CITATION-VERIFICATION GATE: before any citation reaches the user, drop FABRICATED citations (a
        # provably-bogus DOI/arXiv-id that an exact index lookup definitively fails) and MISATTRIBUTED
        # ones (the cited source — external OR corpus — does not actually support the claim). Existence is
        # deterministic + cached; support is a bounded LLM judge; fail-open. The cleaned answer then
        # drives the source panel AND what is saved + cached below, so a wrong citation never reaches the
        # user even on a cache replay.
        answer, _cv_removed = _citeverify.verify_citations(provider, answer=answer, sources=sources)
        _cv_fabricated = _citeverify.has_fabricated(_cv_removed)
        if _cv_removed:
            logger.info("citation gate dropped %d citation(s): %s", len(_cv_removed), _cv_removed)
            yield {"type": "citation_warning", "removed": sorted({n for n, _ in _cv_removed}),
                   "reasons": sorted({reason for _n, reason in _cv_removed}), "n_sources": full_n}
        # Show only the sources that JUSTIFY the answer (the ones it cited): a maths question no
        # longer lists biology hits the search happened to return. Kept sources keep their original
        # number so [n] still resolves; falls back to all when nothing was cited.
        display_sources = _relevant_sources(answer, sources)
        if len(display_sources) != full_n:
            logger.info("source relevance: %d of %d sources cited — showing only those",
                        len(display_sources), full_n)
            yield {"type": "sources", "sources": display_sources}
        _av = mem.add_answer_version(q_version_id, answer, sources=display_sources)

        # Save for reuse ONLY when the generation truly succeeded: provider worked, no
        # exception, the agentic answer passed verification AND its code didn't fail, and
        # the answer wasn't rewritten post-verification. Cache the clean body (no footers).
        dep_verified = (not agentic_loop_enabled()) or (verification_passed(verdict) and not loop_run_failed)
        # The holistic verifier ran BEFORE the citation gate, so its verdict is stale if the gate then
        # removed a FABRICATED (provably-bogus) cited source — a hallucinated reference. Such an answer
        # must not be labeled or cached as 'verified', regardless of the pre-gate score.
        if _cv_fabricated:
            dep_verified = False
        # SELF-CONSISTENT != VERIFIED: confirm by an INDEPENDENT route (re-derive + unit / magnitude /
        # limiting-case sanity) before trusting or caching as verified. A disagreement -> honest confidence.
        ind_check: Dict[str, Any] = {"agrees": None}
        if (dep_verified and provider_ok and not gen_failed and independent_verify_enabled()
                and (answer or "").strip() and answer != "(no answer)"):
            yield {"type": "status",
                   "message": "Independently re-deriving and sanity-checking the answer..."}
            ind_check = independent_check(provider, question=q, answer=answer)
        verified = is_truly_verified(dep_verified, ind_check, consistent=consistency_ok)
        if dep_verified and ind_check.get("agrees") is False:
            issues = "; ".join(str(x) for x in (ind_check.get("issues") or []))[:200]
            yield {"type": "low_confidence", "message": (
                "An independent re-derivation disagreed with this answer"
                + (f" ({issues})" if issues else "")
                + " — it couldn't be independently confirmed, so treat the key claims with caution.")}
        quality_ok = bool(verified and not answer_rewritten)   # reusable only when independently verified
        # Cache the clean body from the GATED answer (footers stripped), so a cached replay inherits the
        # citation gate. On the caching path the generation succeeded, so this equals the clean stream.
        body = _strip_answer_footers(answer) or (clean_body or "").strip()
        body, _ = repair_citations(body, full_n)
        did_cache = False
        if (cache_on and provider_ok and not gen_failed
                and _cacheable_answer(q, body, display_sources)):
            # Store WITH its quality status: a verified answer is reusable; a low-quality one is recorded
            # (verified=0) but NEVER replayed, and a later verified answer upgrades it.
            mem.cache_answer(
                user_id=user_id,
                session_id=session_id,
                question=q,
                answer=body,
                sources=display_sources,
                embedding=query_emb,
                embedding_meta=query_meta,
                verified=quality_ok,
                logic_version=answer_logic_version(),
            )
            did_cache = True
        _sp.set(cached=did_cache, citations_removed=len(removed_citations))
    trace.set(cached=did_cache, n_sources=len(display_sources)).end()
    if removed_citations:
        logger.info("citation guard: removed out-of-range %s (only %d sources)",
                    removed_citations, full_n)
        yield {"type": "citation_warning", "removed": removed_citations, "n_sources": full_n}
    # LEARN from this run (fail-open): a mistake-lesson when the pipeline CORRECTED itself (arithmetic
    # override / conclusion-matches-work reconcile / auto-review rewrite), a preference-lesson on a
    # verified regeneration; reinforce the recalled lessons that helped produce a verified answer.
    try:
        _experience.capture_outcome(
            mem, user_id=user_id, question=q, answer=answer, corrections=learn_corrections,
            reconciled=learn_reconciled, rewritten=answer_rewritten,
            regenerated=regen_qversion_id is not None, verified=quality_ok,
            query_embedding=query_emb, query_meta=query_meta, logic_version=answer_logic_version())
        if quality_ok:
            mem.reinforce_lessons(_ctx.get("lesson_ids") or [])
    except Exception:                                          # noqa: BLE001 - learning never breaks a turn
        pass
    # GROW THE RAG (Phase 2, fail-open): ingest the external findings this VERIFIED answer actually CITED
    # into the grown corpus, embedded for future recall. Scheduled in the BACKGROUND (the embedding call
    # is the one slow step) so it adds ZERO latency to this answer.
    _acquired.capture_findings(
        mem, user_id=user_id, question=q, items=items, cited_sources=display_sources,
        verified=quality_ok, logic_version=answer_logic_version())
    yield {"type": "done", "answer": answer, **_ans_meta(_av)}
