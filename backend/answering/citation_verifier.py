"""
citation_verifier.py — a deterministic citation-verification gate (runs at citation time).

Before any citation reaches the user, every cited source is checked two ways and bad citations are
removed from the answer:

  1. EXISTENCE (deterministic): if the source carries a DOI or an arXiv id, confirm it actually exists
     in a real index (Crossref/OpenAlex/arXiv via `biblio_lookup`). A provably-bogus identifier (an
     exact-id lookup that DEFINITIVELY fails) is FABRICATED -> remove. No id / title-only / a transient
     lookup error is 'unresolvable' -> kept (advisory; never block a legitimately-unindexed source).
     Web pages (a real fetched URL) and local corpus chunks exist by construction -> no lookup.

  2. SUPPORT (LLM judge, extends the relevance gate from "relevant to the question" to "supports THIS
     claim"): for each (claim sentence, cited source) pair, judge whether the source's text genuinely
     supports that specific claim — not merely the same topic. A source that doesn't support the claim
     is MISATTRIBUTED -> remove. Applies to external AND corpus citations (so an unrelated corpus paper
     can't be cited for a general-knowledge fact).

Code-aware: citations inside fenced/inline code are NEVER parsed or rewritten (a matrix literal like
`[[1, 0], [0, 1]]` or an index `arr[0]` is not a citation). "No citation is better than a wrong one":
a removed citation simply leaves the claim uncited; a fully stripped answer is handled by the existing
cites-zero -> reasoning handoff. Deterministic + cached + bounded + 429-safe existence; fail-open
everywhere (any error keeps the answer as drafted). Gated by CITATION_VERIFICATION (default on).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Sentence/segment split that PRESERVES its delimiters (even index = sentence, odd = delimiter), so one
# sentence's citations can be rewritten and re-joined without disturbing the rest of the answer.
_SPLIT_RE = re.compile(r"((?:\n+)|(?:(?<=[.!?])[\"')\]]?\s+))")
# A citation token [n] / [n, m] — but ONLY when the '[' is not glued to an identifier, digit, or another
# bracket. So "method [1]" matches, while "arr[0]", "x[1]", and the inner "[1, 0]" of a matrix "[[1,0]]"
# do NOT. Combined with code-masking below, code/math literals are never treated as citations.
_CITE_RE = re.compile(r"(?<![\w\[\]])\[(\d+(?:\s*,\s*\d+)*)\](?!\d)")
_NUM_RE = re.compile(r"\d+")
_CODE_SPAN_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)   # fenced + inline code -> opaque
_PLACEHOLDER_RE = re.compile(r"\x00C(\d+)\x00")

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_ARXIV_PATH_RE = re.compile(r"/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

_SNIPPET_CHARS = 320
_MAX_CLAIM_CHARS = 240
_SUPPORT_MAX_TOKENS = 220

_SUPPORT_SYSTEM = (
    "You are a STRICT citation-faithfulness judge for a research assistant. You receive numbered CLAIMS, "
    "each with the SOURCE excerpts cited for it. For each (claim, source) pair, decide whether THAT "
    "SOURCE'S TEXT genuinely SUPPORTS THAT SPECIFIC CLAIM — states it, or directly backs it — not merely "
    "that the source is on the same broad topic. A source about the general subject that does not "
    "actually support the claim is NOT support; a source that supports a DIFFERENT claim is NOT support "
    "for this one. Reply with ONLY one line of strict JSON, no prose:\n"
    '{"supported": [[claim_number, source_number], ...]}\n'
    "listing every pair where the source genuinely supports the claim. If none, reply "
    '{"supported": []}.'
)


def enabled() -> bool:
    return os.getenv("CITATION_VERIFICATION", "true").strip().lower() not in ("0", "false", "no", "off")


def _lookup_workers() -> int:
    try:
        return max(1, min(8, int(os.getenv("CITATION_LOOKUP_WORKERS", "4"))))
    except (TypeError, ValueError):
        return 4


# ---------------------------------------------------------------------------
# Identifiers + existence
# ---------------------------------------------------------------------------
def _host(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).netloc or "").lower().rsplit("@", 1)[-1].split(":")[0]
    except Exception:                                  # noqa: BLE001
        return ""


def extract_identifier(source: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """('doi', doi) / ('arxiv', id) for a paper with a CANONICAL identifier — an explicit `doi` field, or
    a URL whose HOST is exactly arxiv.org / doi.org / dx.doi.org. Anything else (a normal web page, a URL
    that merely contains a doi/arxiv-looking substring in a path or query) returns (None, None): its
    existence is implicit (a real fetched URL / a local corpus chunk) and is never index-checked, so a
    real source is never wrongly flagged as a non-existent paper."""
    if not isinstance(source, dict):
        return None, None
    doi = (source.get("doi") or "").strip()
    if doi:
        m = _DOI_RE.search(doi)
        if m:
            return "doi", m.group(0)
    url = source.get("url") or ""
    host = _host(url)
    if host in ("arxiv.org", "www.arxiv.org"):
        m = _ARXIV_PATH_RE.search(urllib.parse.urlparse(url).path)
        if m:
            return "arxiv", m.group(1)
    if host in ("doi.org", "dx.doi.org"):
        m = _DOI_RE.search(url)
        if m:
            return "doi", m.group(0)
    return None, None


def verify_existence(sources: List[Dict[str, Any]], cited: Set[int]) -> Dict[int, str]:
    """Per cited source number -> 'exists' | 'not_found' | 'unresolvable'. Deterministic index lookups
    (cached), bounded + fail-open: a source with no canonical id is 'exists' (implicit), a transient
    lookup error is 'unresolvable'. Only a definitive exact-id 404 becomes 'not_found' (fabricated)."""
    out: Dict[int, str] = {}
    todo: List[Tuple[int, str, str]] = []
    for n in cited:
        src = sources[n - 1] if 1 <= n <= len(sources) else {}
        kind, ident = extract_identifier(src)
        if not ident:
            out[n] = "exists"                          # web page / corpus chunk -> exists by construction
        else:
            todo.append((n, kind, ident))
    if not todo:
        return out

    from backend.external_search import biblio_lookup
    from backend.common.request_context import ContextThreadPoolExecutor

    def _check(item: Tuple[int, str, str]) -> Tuple[int, str]:
        n, kind, ident = item
        try:
            res = biblio_lookup.doi_exists(ident) if kind == "doi" else biblio_lookup.arxiv_id_exists(ident)
        except Exception:                              # noqa: BLE001 - lookup error -> unresolvable
            res = None
        if res is True:
            return n, "exists"
        if res is False:
            return n, "not_found"
        return n, "unresolvable"

    try:
        with ContextThreadPoolExecutor(max_workers=_lookup_workers()) as ex:
            for n, status in ex.map(_check, todo):
                out[n] = status
    except Exception:                                  # noqa: BLE001 - never let the gate break the answer
        for n, _k, _i in todo:
            out.setdefault(n, "unresolvable")
    return out


# ---------------------------------------------------------------------------
# Code masking + claims
# ---------------------------------------------------------------------------
def _mask_code(answer: str) -> Tuple[str, List[str]]:
    """Replace fenced/inline code spans with opaque placeholders so the gate never parses or rewrites
    citations inside code/math. Returns (masked_text, spans)."""
    spans: List[str] = []

    def _sub(m: "re.Match") -> str:
        spans.append(m.group(0))
        return f"\x00C{len(spans) - 1}\x00"

    return _CODE_SPAN_RE.sub(_sub, answer), spans


def _unmask_code(masked: str, spans: List[str]) -> str:
    def _sub(m: "re.Match") -> str:
        i = int(m.group(1))
        return spans[i] if 0 <= i < len(spans) else m.group(0)

    return _PLACEHOLDER_RE.sub(_sub, masked)


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(" ", text)


def _find_cites(text: str) -> Set[int]:
    nums: Set[int] = set()
    for m in _CITE_RE.finditer(text or ""):
        nums.update(int(x) for x in _NUM_RE.findall(m.group(1)))
    return nums


def claims_with_citations(answer: str) -> List[Tuple[int, str, Set[int]]]:
    """(segment_index, sentence_text, cited_numbers) for each cited sentence (code spans excluded). The
    segment_index indexes the masked `_SPLIT_RE` segments so a caller can rewrite exactly that sentence;
    the text is returned unmasked for display."""
    masked, _spans = _mask_code(answer)
    segs = _SPLIT_RE.split(masked)
    out: List[Tuple[int, str, Set[int]]] = []
    for i in range(0, len(segs), 2):
        nums = _find_cites(segs[i])
        if nums:
            out.append((i, _strip_placeholders(segs[i]).strip(), nums))
    return out


def _excerpt(sources: List[Dict[str, Any]], n: int) -> str:
    src = sources[n - 1] if 1 <= n <= len(sources) else {}
    return (src.get("text") or src.get("chunk_text") or "").strip()


def _source_excerpt(sources: List[Dict[str, Any]], n: int) -> str:
    src = sources[n - 1] if 1 <= n <= len(sources) else {}
    title = (src.get("title") or "Untitled").strip()
    text = re.sub(r"\s+", " ", _excerpt(sources, n))[:_SNIPPET_CHARS]
    return f"{title} — {text}" if text else title


def _parse_supported(raw: str) -> Optional[Set[Tuple[int, int]]]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("supported"), list):
        return None
    out: Set[Tuple[int, int]] = set()
    for pair in obj["supported"]:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            try:
                out.add((int(pair[0]), int(pair[1])))
            except (TypeError, ValueError):
                continue
    return out


def verify_support(provider, claims: List[Tuple[int, str, Set[int]]],
                   sources: List[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    """The set of (segment_index, source_number) pairs the cited source GENUINELY supports. Fail-OPEN:
    when the provider is unavailable or the verdict can't be parsed, treat ALL cited pairs as supported
    (keep). A cited source with NO excerpt text is also kept (we can't fairly judge it from a title)."""
    all_pairs = {(sidx, n) for (sidx, _t, nums) in claims for n in nums}
    if not all_pairs:
        return set()
    no_text = {(sidx, n) for (sidx, _t, nums) in claims for n in nums if not _excerpt(sources, n)}
    judgeable = all_pairs - no_text
    if not judgeable or provider is None or not getattr(provider, "is_available", False):
        return all_pairs                               # nothing to judge / no provider -> keep all

    lines: List[str] = []
    claim_map: Dict[int, int] = {}                     # claim_number (1-based, shown) -> segment_index
    k = 0
    for (sidx, text, nums) in claims:
        judged_nums = sorted(n for n in nums if (sidx, n) in judgeable)
        if not judged_nums:
            continue
        k += 1
        claim_map[k] = sidx
        clean = re.sub(r"\s+", " ", _strip_placeholders(text).strip())[:_MAX_CLAIM_CHARS]
        lines.append(f'CLAIM {k}: "{clean}"')
        for n in judged_nums:
            lines.append(f"  [{n}] {_source_excerpt(sources, n)}")
    user = "\n".join(lines)
    try:
        parts: List[str] = []
        for tok in provider.stream_chat(
            [{"role": "user", "content": user}],
            system=_SUPPORT_SYSTEM, max_tokens=_SUPPORT_MAX_TOKENS, temperature=0.0,
        ):
            if isinstance(tok, str):
                parts.append(tok)
        raw = "".join(parts)
    except Exception:                                  # noqa: BLE001 - provider error -> fail-open
        return all_pairs

    parsed = _parse_supported(raw)
    if parsed is None:                                 # unparseable -> fail-open (keep all)
        return all_pairs
    supported: Set[Tuple[int, int]] = set(no_text)     # empty-excerpt sources are kept (advisory)
    for (claim_no, n) in parsed:
        sidx = claim_map.get(claim_no)
        if sidx is not None and (sidx, n) in judgeable:
            supported.add((sidx, n))
    return supported


# ---------------------------------------------------------------------------
# Top-level gate
# ---------------------------------------------------------------------------
def _rewrite_segment(text: str, drop: Set[int]) -> str:
    """Drop the citation numbers in `drop` from this ONE segment's citation tokens, then tidy only this
    segment (so whitespace elsewhere — code, tables — is never touched)."""
    def _fix(match: "re.Match") -> str:
        kept = [s for s in _NUM_RE.findall(match.group(1)) if int(s) not in drop]
        return "[" + ", ".join(kept) + "]" if kept else ""

    out = _CITE_RE.sub(_fix, text)
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)       # " word  ." -> " word."
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


def verify_citations(provider, *, answer: str,
                     sources: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, str]]]:
    """Drop FABRICATED (provably-bogus id) and MISATTRIBUTED (source doesn't support the claim) citations
    from `answer`. Returns (clean_answer, removed) where removed is [(source_number, reason)]. Code spans
    are never touched. Fail-open: any error returns the answer unchanged. No-op when disabled or nothing
    is cited."""
    if not enabled() or not (answer or "").strip():
        return answer, []
    try:
        masked, spans = _mask_code(answer)
        cited = {n for n in _find_cites(masked) if 1 <= n <= len(sources)}
        if not cited:
            return answer, []
        existence = verify_existence(sources, cited)
        segs = _SPLIT_RE.split(masked)
        claims: List[Tuple[int, str, Set[int]]] = []
        for i in range(0, len(segs), 2):
            nums = _find_cites(segs[i])
            if nums:
                claims.append((i, segs[i], nums))
        supported = verify_support(provider, claims, sources)

        removed: List[Tuple[int, str]] = []
        for (sidx, _text, nums) in claims:
            drop: Set[int] = set()
            for n in nums:
                if existence.get(n) == "not_found":
                    drop.add(n)
                    removed.append((n, "fabricated"))
                elif (sidx, n) not in supported:
                    drop.add(n)
                    removed.append((n, "misattributed"))
            if drop:
                segs[sidx] = _rewrite_segment(segs[sidx], drop)
        if not removed:
            return answer, []
        return _unmask_code("".join(segs), spans), removed
    except Exception:                                  # noqa: BLE001 - the gate must never break an answer
        logger.warning("citation verification failed; keeping the answer as drafted", exc_info=True)
        return answer, []
