"""
conversation_router.py  --  Decide whether a chat message is a conversational FOLLOW-UP
(answerable from the conversation so far, possibly by re-running code from an earlier turn) or a
NEW, self-contained question that needs retrieval.

The chat pipeline otherwise treats EVERY message as a fresh research topic: it plans sub-question
"angles" and runs the full web/arXiv/patent/GitHub sweep. For a follow-up like
"what is the output of the above code?" that is pure waste — it pulls dozens of off-topic sources
and never uses the conversation. This router fixes that by classifying the message first.

It returns a `ConversationRoute`:
    kind : str
        "code_output"  -- asks for the OUTPUT/result of code or work from an earlier turn
                          -> re-run that code in the sandbox; no search.
        "context"      -- a follow-up answerable from the conversation (anaphora: "above", "it",
                          "that", elliptical) -> answer from the chat; no search.
        "research"     -- a self-contained question that needs retrieval -> search as usual.
    resolved_query : str
        the message rewritten to stand alone (anaphora resolved), used for retrieval in the
        "research" case and for display. Equal to the question when nothing needs resolving.
    confidence : float
    source : str   -- "llm" | "regex" (provenance, for logs)

Design (mirrors backend.answering.task_classifier):
  * One fast-model LLM call returning strict JSON, timeout-bounded.
  * NOT cached: the same text means different things in different conversations.
  * High safety: defaults to "research" on no history / disabled / any LLM failure, so a genuine
    new question is never starved of retrieval. Only a confident follow-up signal skips search.

Toggle / tune via .env (read live):
    CONVERSATION_ROUTER=true|false   (default true; false = always "research")
    CONVERSATION_ROUTER_TIMEOUT=3.0  (seconds; fall back to regex after this)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

KINDS = ("code_output", "context", "research")

_MAX_TOKENS = 200
_DEFAULT_TIMEOUT = 3.0
_RECENT_TURNS = 6           # how many trailing turns to show the router
_CONTEXT_CHARS = 4000       # cap on the transcript handed to the model


@dataclass(frozen=True)
class ConversationRoute:
    kind: str
    resolved_query: str
    confidence: float = 0.0
    source: str = "regex"


# ----------------------------------------------------------------------
# Config (read live so .env / tests take effect)
# ----------------------------------------------------------------------
def router_enabled() -> bool:
    return os.getenv("CONVERSATION_ROUTER", "true").strip().lower() not in ("0", "false", "no", "off")


def _timeout() -> float:
    try:
        return max(0.3, float(os.getenv("CONVERSATION_ROUTER_TIMEOUT", str(_DEFAULT_TIMEOUT))))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


# ----------------------------------------------------------------------
# Regex fallback (used when the LLM is unavailable)
# ----------------------------------------------------------------------
# EXPLICIT back-references to the conversation (not bare pronouns like "it/this/that", which appear
# in countless self-contained questions). Kept deliberately narrow so the regex fallback almost
# never mislabels a NEW question as a follow-up; the LLM router does the nuanced work, and the
# integration applies a confidence floor on top (the regex verdict scores below it on purpose).
_ANAPHORA_RE = re.compile(
    r"\b("
    r"above|previously|earlier|aforementioned|"
    # "the/that/this <structural reference>" — NOT 'output'/'result', which are content words that
    # appear in plenty of self-contained questions ("the output impedance", "the result of the 2020
    # election"). Only references to the prior turn's artefact count.
    r"(the|that|this) (above |previous |earlier |last |same )?"
    r"(code|answer|reply|snippet|program|script|solution|function|implementation|example)|"
    r"(re-?run|rerun|run|execute) (it|that|this|the (code|program|script))"
    r")\b", re.I)
# Asking for the OUTPUT/result of code/work.
_OUTPUT_RE = re.compile(
    r"\b(output|results?|prints?|printed|returns?|returned|stdout|run it|execute[ds]?|"
    r"what (does|do|will) (it|this|that))\b", re.I)


def _regex_route(question: str, has_history: bool) -> ConversationRoute:
    # Confidence is 0.5 ON PURPOSE: it sits below the integration's follow-up confidence floor, so a
    # regex verdict never on its own diverts a question away from search. It is a hint/log signal;
    # the LLM verdict (higher confidence) is what actually routes a follow-up.
    q = (question or "").strip()
    if not has_history or not q:
        return ConversationRoute("research", q, 0.5, "regex")
    anaphora = bool(_ANAPHORA_RE.search(q))
    short = len(q.split()) <= 12
    if anaphora and _OUTPUT_RE.search(q):
        return ConversationRoute("code_output", q, 0.5, "regex")
    if anaphora and short:
        return ConversationRoute("context", q, 0.5, "regex")
    return ConversationRoute("research", q, 0.5, "regex")


# ----------------------------------------------------------------------
# LLM classification
# ----------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You route the LATEST user message in an ongoing chat with a research assistant. Using the "
    "conversation so far, decide how it should be answered:\n"
    '  "code_output"  -- it asks for the OUTPUT, result, printed/returned value, or behaviour of '
    "code or a computation from an EARLIER turn (e.g. 'what is the output of the above code?', "
    "'what does it print?', 'run that').\n"
    '  "context"      -- it is a follow-up that can be answered from the conversation itself '
    "(refers back with 'this/that/it/the above', is elliptical, asks to explain/expand/rephrase "
    "something already discussed) and does NOT need new external sources.\n"
    '  "research"     -- it is a self-contained NEW question that needs fresh retrieval/search '
    "(names its own topic; does not depend on the earlier turns).\n"
    "Also rewrite the message as a STANDALONE query with all references resolved from the "
    "conversation (e.g. 'explain the above' -> 'explain the MVDR beamformer'); if it is already "
    "standalone, repeat it unchanged.\n"
    "When in doubt, choose \"research\" (a missed follow-up is cheaper than a missing answer).\n"
    "Reply with ONLY one line of strict JSON, no prose:\n"
    '{"kind": "code_output"|"context"|"research", "resolved_query": "...", "confidence": 0.0-1.0}'
)


def _format_recent(history: List[Dict[str, str]], summary: str) -> str:
    parts: List[str] = []
    if summary:
        parts.append("Summary of earlier conversation:\n" + summary)
    for t in history[-_RECENT_TURNS:]:
        role = t.get("role", "user")
        content = (t.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)[-_CONTEXT_CHARS:]


def _parse_json(raw: str) -> Optional[dict]:
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


def _normalize_kind(kind: object) -> str:
    k = str(kind or "").strip().lower()
    return k if k in KINDS else "research"


def _llm_route(question: str, transcript: str) -> Optional[ConversationRoute]:
    """One short, timeout-bounded LLM call. Returns a ConversationRoute or None on
    unavailability / timeout / error / unparseable output (caller falls back to regex)."""
    from backend.llm.streaming_provider import get_provider

    provider = get_provider()
    if not provider.is_available:
        return None

    user = f"CONVERSATION SO FAR:\n{transcript}\n\nLATEST USER MESSAGE:\n{question}"

    def _run() -> str:
        parts: List[str] = []
        total = 0
        for tok in provider.stream_chat(
            [{"role": "user", "content": user}],
            system=_SYSTEM_PROMPT,
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        ):
            if not isinstance(tok, str):
                continue
            parts.append(tok)
            total += len(tok)
            if total > 1500:
                break
        return "".join(parts)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            raw = fut.result(timeout=_timeout())
        except Exception:                               # noqa: BLE001 - timeout/provider error
            return None

    obj = _parse_json(raw)
    if obj is None:
        return None
    kind = _normalize_kind(obj.get("kind"))
    resolved = str(obj.get("resolved_query") or "").strip() or question
    try:
        # Default BELOW the integration's follow-up floor: if the model omits a confidence, do NOT
        # divert away from search on its say-so alone.
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return ConversationRoute(kind, resolved[:600], conf, "llm")


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def route(question: Optional[str], history: Optional[List[Dict[str, str]]] = None,
          *, summary: str = "") -> ConversationRoute:
    """Classify the latest message given the prior conversation. Never raises; falls back to the
    regex verdict (and to 'research') on no history / disabled / any LLM failure."""
    q = (question or "").strip()
    hist = list(history or [])
    has_history = bool(hist)
    regex = _regex_route(q, has_history)
    try:
        if not q or not has_history or not router_enabled():
            return regex
        transcript = _format_recent(hist, summary)
        llm = _llm_route(q, transcript)
        if llm is None:
            return regex                                # transient failure -> regex verdict
        return llm
    except Exception:                                   # noqa: BLE001 - never break routing
        return regex


def is_followup(route_result: ConversationRoute) -> bool:
    """Convenience: the message should be answered from the conversation, not a fresh search."""
    return route_result.kind in ("code_output", "context")
