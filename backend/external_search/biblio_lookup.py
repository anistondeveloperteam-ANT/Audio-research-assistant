"""
biblio_lookup.py — deterministic EXISTENCE checks against real bibliographic indexes.

Given a DOI or an arXiv ID, confirm the work actually exists by an exact lookup in a real index:
DOI -> Crossref (then OpenAlex as an independent fallback); arXiv ID -> the arXiv API. Returns:

    True   the index has it (exists),
    False  the index DEFINITIVELY does not (an exact-id 404) -> the identifier is fabricated,
    None   transient error / can't tell -> 'unresolvable' (advisory; never auto-remove on this).

Deterministic (real index lookups, not an LLM guess), cached on disk so a repeat lookup makes no
network call (reuses the external-search cache), and 429/5xx-backed-off. Key-free APIs only; no new
dependency (uses `requests`, already used across external_search).
"""
from __future__ import annotations

import logging
import re
import time
import urllib.parse
from typing import Optional

import requests

from backend.external_search.base import USER_AGENT, cache_get, cache_set

logger = logging.getLogger(__name__)

_TIMEOUT = 8.0
_RETRIES = 2
_CACHE_TTL = 60 * 60 * 24 * 30        # 30 days — a work's existence is stable


def _status_get(url: str, *, accept: str = "application/json") -> Optional[int]:
    """GET `url`, return the HTTP status code, or None on a transient/network failure. 429/5xx are
    retried with exponential backoff so a rate limit never reads as 'not found'."""
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    for attempt in range(_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=_TIMEOUT, allow_redirects=True)
        except requests.RequestException:
            return None
        code = resp.status_code
        if code == 429 or 500 <= code < 600:           # transient -> back off and retry
            if attempt < _RETRIES:
                time.sleep(min(2 ** attempt, 8))
                continue
            return None
        return code
    return None


def doi_exists(doi: str) -> Optional[bool]:
    """True if the DOI resolves in Crossref or OpenAlex; False only if BOTH definitively 404; None if we
    cannot tell (transient). Cached by DOI."""
    d = (doi or "").strip().strip(".")
    if not d:
        return None
    key = f"biblio:doi:{d.lower()}"
    hit = cache_get(key, _CACHE_TTL)
    if isinstance(hit, dict) and "exists" in hit:
        return hit["exists"]
    result = _doi_lookup(d)
    if result is not None:                             # cache only a definitive True/False answer
        cache_set(key, {"exists": result})
    return result


def _doi_lookup(doi: str) -> Optional[bool]:
    enc = urllib.parse.quote(doi, safe="")
    code = _status_get(f"https://api.crossref.org/works/{enc}")
    if code == 200:
        return True
    cross_404 = (code == 404)
    code2 = _status_get(f"https://api.openalex.org/works/doi:{enc}")   # independent index fallback
    if code2 == 200:
        return True
    if cross_404 and code2 == 404:
        return False                                   # both indexes lack it -> fabricated
    return None                                        # at least one was transient -> unresolvable


def arxiv_id_exists(arxiv_id: str) -> Optional[bool]:
    """True if the arXiv API returns the paper; False if it definitively has none; None on transient
    error. Cached by id."""
    a = (arxiv_id or "").strip()
    if not a:
        return None
    key = f"biblio:arxiv:{a.lower()}"
    hit = cache_get(key, _CACHE_TTL)
    if isinstance(hit, dict) and "exists" in hit:
        return hit["exists"]
    result = _arxiv_lookup(a)
    if result is not None:
        cache_set(key, {"exists": result})
    return result


def _arxiv_lookup(arxiv_id: str) -> Optional[bool]:
    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}&max_results=1"
    for attempt in range(_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=_TIMEOUT)
        except requests.RequestException:
            return None
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt < _RETRIES:
                time.sleep(min(2 ** attempt, 8))
                continue
            return None
        if resp.status_code != 200:
            return None
        body = resp.text or ""
        if "<title>Error</title>" in body:             # arXiv's error feed -> the id is not a real paper
            return False
        m = re.search(r"<opensearch:totalResults[^>]*>(\d+)</", body)
        if m:                                          # the API's own count is authoritative
            return int(m.group(1)) > 0
        base = arxiv_id.split("v")[0]
        return (f"arxiv.org/abs/{base}" in body) or (f"<id>http://arxiv.org/abs/{arxiv_id}" in body)
    return None
