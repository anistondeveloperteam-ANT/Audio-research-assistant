"""
relevance_gate.py — Source-relevance gate.

Retrieval can return sources that share the question's broad TOPIC but do not actually address
what was asked (e.g. audio papers retrieved for an audio-storage *calculation*). Reranker scores
cannot tell those apart (high topical similarity, wrong content), so before any retrieved source is
allowed to ground or be cited in the answer, a bounded LLM judge confirms it DIRECTLY helps answer
THIS question. Sources it rejects are dropped; if it rejects them all, the caller answers from
reasoning with no citation.

Fail-OPEN by design: on disabled / unavailable provider / empty / unparseable output / any error,
keep ALL sources — a transient hiccup must never silently strip grounding. Gated by
SOURCE_RELEVANCE_GATE (default on). No new dependency.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

_MAX_ITEMS = 12          # judge at most the top-N sources (token bound); unseen ones are kept
_SNIPPET_CHARS = 320     # per-source excerpt shown to the judge
_MAX_TOKENS = 60         # the verdict is a short JSON list

_RELEVANCE_SYSTEM = (
    "You are a STRICT relevance filter for a research assistant. You receive a QUESTION and a "
    "numbered list of retrieved SOURCE excerpts. Decide which sources contain information that "
    "DIRECTLY helps answer THIS specific question — not merely sources on the same broad topic. "
    "A source that is about the general subject but does NOT address what the question actually "
    "asks is NOT relevant. Be strict: if a source would not let you support, ground, or improve "
    "the answer to this exact question, EXCLUDE it.\n"
    "Honour the question's SCOPE and RECENCY: if the question names a time frame, version, place, "
    "entity, or other constraint, EXCLUDE sources that fall outside it (e.g. a different year/era, an "
    "older version, a different entity) even when they are on-topic — out-of-scope material must not be "
    "presented as if it answers the question. Judge each source only on whether it addresses the "
    "question within its scope, never on how many you keep.\n"
    "Reply with ONLY one line of strict JSON, no prose:\n"
    '{"relevant": [the source numbers that genuinely address the question]}\n'
    'If NONE genuinely address the question, reply {"relevant": []}.'
)


def relevance_gate_enabled() -> bool:
    """Live env read so .env / tests take effect without a redeploy (default ON)."""
    return os.getenv("SOURCE_RELEVANCE_GATE", "true").strip().lower() not in (
        "0", "false", "no", "off")


def _snippet(item: Dict[str, Any]) -> str:
    title = (item.get("title") or "Untitled").strip()
    text = re.sub(r"\s+", " ", (item.get("text") or item.get("chunk_text") or "").strip())
    text = text[:_SNIPPET_CHARS]
    return f"{title} — {text}" if text else title


def _all_indices(n: int) -> Set[int]:
    return set(range(1, n + 1))


def _parse_relevant(raw: str, n: int) -> Optional[Set[int]]:
    """Parse {"relevant": [...]} into valid 1-based indices. Returns None when NO verdict can be
    parsed (the caller treats None as fail-open = keep all); an empty set means the judge genuinely
    found none relevant."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or "relevant" not in obj:
        return None
    rel = obj.get("relevant")
    if not isinstance(rel, list):
        return None
    out: Set[int] = set()
    for v in rel:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n:
            out.add(i)
    return out                       # may be empty -> genuinely "none relevant"


def relevant_source_indices(provider, *, question: str, items: List[Dict[str, Any]],
                            max_items: int = _MAX_ITEMS) -> Set[int]:
    """The 1-based indices of the sources that genuinely address `question`.

    Fail-OPEN: returns ALL indices when the gate is disabled, the provider is unavailable, or no
    verdict can be parsed. Sources beyond `max_items` are not judged and are always kept."""
    n = len(items)
    if n == 0:
        return set()
    if not relevance_gate_enabled():
        return _all_indices(n)
    if provider is None or not getattr(provider, "is_available", False):
        return _all_indices(n)

    judged = items[:max_items]
    listing = "\n".join(f"[{i}] {_snippet(it)}" for i, it in enumerate(judged, 1))
    user = f"QUESTION:\n{question}\n\nSOURCES:\n{listing}"
    try:
        parts: List[str] = []
        for tok in provider.stream_chat(
            [{"role": "user", "content": user}],
            system=_RELEVANCE_SYSTEM, max_tokens=_MAX_TOKENS, temperature=0.0,
        ):
            if isinstance(tok, str):
                parts.append(tok)
        raw = "".join(parts)
    except Exception:                # noqa: BLE001 - any provider error -> fail-open (keep all)
        return _all_indices(n)

    verdict = _parse_relevant(raw, len(judged))
    if verdict is None:              # unparseable -> fail-open
        return _all_indices(n)
    if n > len(judged):              # never drop a source we didn't judge
        verdict |= set(range(len(judged) + 1, n + 1))
    return verdict
