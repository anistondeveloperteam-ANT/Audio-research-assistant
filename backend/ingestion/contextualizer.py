"""
Anthropic-style Contextual Retrieval.

For each chunk we ask the configured LLM (Gemini, via get_provider()) for one short sentence that
situates the chunk within its document. That sentence is stored separately (the `context_text`
column) and prepended to the chunk ONLY for indexing — the original chunk text is unchanged and is
what users see in citations. The contextual prefix measurably improves vector + BM25 recall.

Production-safe:
  - Off-switch: CONTEXTUAL_CHUNKS=false → callers get empty contexts and fall back to plain chunks.
  - Fail-safe: any LLM error / unavailable provider → "" (plain chunk), never raises.
  - Cheap re-runs: generated contexts are cached on disk keyed by (document, chunk).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = ROOT / "data" / "extracted" / "contextual_cache.json"

_PROMPT = (
    "<document>\n{doc}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context (one sentence, roughly 50-100 tokens) to situate this chunk "
    "within the overall document, to improve search retrieval of the chunk. "
    "Answer ONLY with the succinct context and nothing else."
)
_SYSTEM = "You write one concise context sentence to help document search. Output the sentence only."

# Batched prompt: situate MANY chunks in ONE LLM call (≈N/batch_size calls instead of one per chunk).
# The model returns a JSON array of exactly N sentences in chunk order; we parse it back per chunk.
_BATCH_PROMPT = (
    "<document>\n{doc}\n</document>\n\n"
    "Below are {n} chunks from the document, each marked with [CHUNK i]. For EACH chunk, write a short, "
    "succinct context (one sentence, ~50-100 tokens) that situates THAT chunk within the overall "
    "document, to improve search retrieval of that chunk.\n\n"
    "{chunks}\n\n"
    "Answer with ONLY a JSON array of exactly {n} strings — one context sentence per chunk, in the same "
    "order as the chunks — and nothing else. Example: [\"context for chunk 1\", \"context for chunk 2\"]"
)
_BATCH_SYSTEM = ("You write one concise context sentence per chunk to help document search. "
                 "Output ONLY a JSON array of strings, one per chunk, in order.")


def contextual_enabled() -> bool:
    """LLM-written situating sentences are OPT-IN (default OFF). They add a small retrieval-recall
    boost but cost one LLM call per ~8 chunks (slow + rate-limit-prone on free tiers). The instant
    'contextual chunk header' (paper title + section, added at embed time) already situates every
    chunk for free, so the fast default is OFF. Set CONTEXTUAL_CHUNKS=true for the extra few %."""
    return os.getenv("CONTEXTUAL_CHUNKS", "false").strip().lower() == "true"


def _doc_chars() -> int:
    return int(os.getenv("CONTEXTUAL_DOC_CHARS", "12000"))


def _max_tokens() -> int:
    # Generous: reasoning models (Gemini 2.5) spend tokens "thinking", which would otherwise
    # starve the short answer. The situating sentence itself is only ~50-100 tokens.
    return int(os.getenv("CONTEXTUAL_MAX_TOKENS", "1024"))


def _retries() -> int:
    return max(1, int(os.getenv("CONTEXTUAL_RETRIES", "3")))


def _batch_size() -> int:
    """How many chunks to situate in ONE LLM call. ~8 turns a 75-chunk paper from 75 calls into ~10 —
    much faster and far gentler on provider rate limits. 1 = the old one-call-per-chunk behavior."""
    try:
        return max(1, int(os.getenv("CONTEXTUAL_BATCH_SIZE", "8")))
    except (TypeError, ValueError):
        return 8


def _batch_max_tokens(n: int) -> int:
    """Output budget for an n-chunk batch: room for n sentences PLUS a reasoning model's thinking, so
    the JSON array isn't truncated (a truncated reply just triggers the safe per-chunk fallback)."""
    return _max_tokens() + max(0, n) * 200


def _concurrency() -> int:
    """How many situating-sentence LLM calls to run at once (cache-miss chunks). The dominant cost
    of ingestion is one call per chunk; running them concurrently cuts a 40-chunk paper from minutes
    to seconds. 1 = serial (no threads). Lower it if your provider's rate limit complains."""
    try:
        return max(1, int(os.getenv("CONTEXTUAL_CONCURRENCY", "6")))
    except (TypeError, ValueError):
        return 6


def _cache_key(doc_text: str, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update((doc_text or "")[: _doc_chars()].encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update((chunk_text or "").encode("utf-8", "ignore"))
    return h.hexdigest()


def _load_cache() -> Dict[str, str]:
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict[str, str]) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _contextual_model_names():
    """Ordered model preference for contextualization. A concurrency-safe PRIMARY first, then a
    cross-vendor fallback, then the chat default as a last resort.

    Contextualization fires a BURST of concurrent calls per paper (one per chunk), so a fast-but-
    rate-limited model (e.g. Mistral, which 429s under the burst) belongs LATER, and a concurrency-
    tolerant model (Gemini) belongs FIRST — otherwise the whole paper's contexts come back empty.
    All overridable: CONTEXTUAL_MODEL (primary), CONTEXTUAL_FALLBACK_MODELS (comma-list)."""
    primary = (os.getenv("CONTEXTUAL_MODEL", "gemini-2.5-flash") or "").strip()
    names = [primary] if primary else []
    names += [m.strip() for m in os.getenv("CONTEXTUAL_FALLBACK_MODELS", "mistral-large-latest").split(",")]
    names.append(None)                                  # the configured chat default, as a final fallback
    return names


def _provider_chain():
    """Available contextualization providers, in preference order (deduped). Cross-vendor, so a single
    provider's rate-limit / quota (429) can't blank the whole paper's contexts."""
    from backend.llm.streaming_provider import get_provider
    out, seen = [], set()
    for name in _contextual_model_names():
        if name is not None and not name:
            continue
        try:
            p = get_provider(name) if name else get_provider()
        except Exception:
            continue
        mid = getattr(p, "model", name)
        if mid in seen or not getattr(p, "is_available", False):
            continue
        seen.add(mid)
        out.append(p)
    return out


def _complete(provider, prompt: str, *, system: str = _SYSTEM, max_tokens: Optional[int] = None) -> str:
    """One LLM call against a single provider. Returns "" on error/empty/unavailable (never raises).
    Skips reasoning dicts; keeps answer text only. For a batch call, the raw text is returned UN-squashed
    (newlines kept) so the JSON array survives parsing."""
    try:
        if not getattr(provider, "is_available", False):
            return ""
        parts: List[str] = []
        for piece in provider.stream_chat(
            [{"role": "user", "content": prompt}],
            system=system, max_tokens=max_tokens or _max_tokens(), temperature=0.0,
        ):
            if isinstance(piece, str):
                parts.append(piece)
        return "".join(parts).strip()
    except Exception:
        return ""


def _run_chain(chain: list, prompt: str, *, system: str = _SYSTEM, max_tokens: Optional[int] = None) -> str:
    """Try each provider in order; on success promote the winner so later chunks skip dead models.
    Retries the whole chain with backoff for transient rate-limit / empty replies."""
    if not chain:
        return ""
    for attempt in range(_retries()):
        for i, prov in enumerate(list(chain)):
            out = _complete(prov, prompt, system=system, max_tokens=max_tokens)
            if out:
                if i > 0:
                    chain.insert(0, chain.pop(i))   # prefer the working model next time
                return out
        if attempt < _retries() - 1:
            time.sleep(min(2 ** attempt, 20))
    return ""


def generate_context(doc_text: str, chunk_text: str, provider=None) -> str:
    """One situating sentence for `chunk_text`. Returns "" on any failure (caller falls back to the
    plain chunk). With no `provider`, tries the configured model then the fallback model(s).
    `provider` is injectable for tests (single provider, single attempt)."""
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return ""
    prompt = _PROMPT.format(doc=(doc_text or "")[: _doc_chars()], chunk=chunk_text[:4000])
    if provider is not None:
        return _complete(provider, prompt)
    try:
        chain = _provider_chain()
    except Exception:
        return ""
    return _run_chain(chain, prompt)


def _parse_batch_response(text: str, n: int) -> Optional[List[str]]:
    """Parse a JSON array of EXACTLY `n` context strings from a batch reply. Tolerant of ```json fences
    and stray prose around the array. Returns the list, or None if it can't get exactly n strings (the
    caller then falls back to per-chunk calls, so correctness is never lost)."""
    if not text or n <= 0:
        return None
    s = text.strip()
    if s.startswith("```"):                                  # strip ```json ... ``` fences
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    start, end = s.find("["), s.rfind("]")                   # the outermost JSON array
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        arr = json.loads(s[start:end + 1])
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    out: List[str] = []
    for item in arr:
        if not isinstance(item, str):
            return None
        out.append(" ".join(item.split()).strip())          # tidy each context to one clean line
    return out


def _contextualize_batch(chain: list, doc_prefix: str, items: list) -> list:
    """Situate a BATCH of cache-miss chunks in ONE LLM call. `items` = [(idx, ctext, key), ...]; returns
    [(idx, key, ctx), ...]. Falls back to one-call-per-chunk if the batched reply can't be parsed into
    exactly len(items) contexts — so accuracy is never worse than the per-chunk path."""
    if not items:
        return []
    if len(items) == 1:                                      # nothing to batch
        idx, ctext, key = items[0]
        return [(idx, key, _run_chain(chain, _PROMPT.format(doc=doc_prefix, chunk=ctext[:4000])))]
    numbered = "\n\n".join(f"[CHUNK {i + 1}]\n{ctext[:4000]}" for i, (_, ctext, _) in enumerate(items))
    prompt = _BATCH_PROMPT.format(doc=doc_prefix, n=len(items), chunks=numbered)
    raw = _run_chain(chain, prompt, system=_BATCH_SYSTEM, max_tokens=_batch_max_tokens(len(items)))
    parsed = _parse_batch_response(raw, len(items)) if raw else None
    if parsed is not None:
        return [(items[i][0], items[i][2], parsed[i]) for i in range(len(items))]
    # fallback: per-chunk (correctness preserved even if the batch reply was malformed)
    out = []
    for idx, ctext, key in items:
        out.append((idx, key, _run_chain(list(chain), _PROMPT.format(doc=doc_prefix, chunk=ctext[:4000]))))
    return out


def contextualize_chunks(doc_text: str, chunks: List[dict], provider=None) -> List[str]:
    """Return one context string per chunk (same order). Builds the provider chain once, uses the
    on-disk cache, and only calls the LLM for cache misses. Cache-miss chunks are BATCHED — one LLM
    call situates several chunks at once (CONTEXTUAL_BATCH_SIZE, default 8), so a 75-chunk paper is
    ~10 calls instead of 75 — and the batches run CONCURRENTLY. Returns ['']*len when disabled; never
    raises; a malformed batch reply falls back to per-chunk calls so output is unchanged.

    The first BATCH is processed sequentially to promote the working model to the front of the shared
    chain (a quota-exhausted model is then skipped for the rest); the remaining batches run in parallel
    with that promoted order — or serially when CONTEXTUAL_CONCURRENCY=1."""
    n = len(chunks)
    if not contextual_enabled() or n == 0:
        return [""] * n
    chain = [provider] if provider is not None else _provider_chain()
    cache = _load_cache()
    out: List[str] = [""] * n
    doc_prefix = (doc_text or "")[: _doc_chars()]

    misses = []                       # (index, chunk_text, cache_key) for chunks not already cached
    for idx, ch in enumerate(chunks):
        ctext = (ch.get("text") or "").strip()
        if not ctext:
            continue
        key = _cache_key(doc_text, ctext)
        if key in cache:
            out[idx] = cache[key]
        else:
            misses.append((idx, ctext, key))

    if not (misses and chain):
        return out

    # Group cache-miss chunks into batches — one LLM call per batch (not per chunk).
    bsize = _batch_size()
    batches = [misses[i:i + bsize] for i in range(0, len(misses), bsize)]

    def _store(results):
        for idx, key, ctx in results:
            out[idx] = ctx
            cache[key] = ctx

    # 1) Warm up on the FIRST batch sequentially against the SHARED chain, so _run_chain's "promote the
    #    working provider" persists; the parallel batches below then try that provider first.
    _store(_contextualize_batch(chain, doc_prefix, batches[0]))

    # 2) Remaining batches: parallel (default) or serial when concurrency is 1 (also the test path).
    rest = batches[1:]
    workers = max(1, min(_concurrency(), len(rest))) if rest else 1
    if rest and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda b: _contextualize_batch(list(chain), doc_prefix, b), rest))
    else:
        results = [_contextualize_batch(list(chain), doc_prefix, b) for b in rest]
    for batch_result in results:
        _store(batch_result)

    _save_cache(cache)
    return out
