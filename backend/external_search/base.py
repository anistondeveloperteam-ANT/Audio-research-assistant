"""
Foundation for external search: the shared `ExternalSource` data structure,
basic URL validation, an HTTP fetcher with timeouts / size caps / retries, and a
tiny TTL disk cache. Everything else in this package builds on it.

Fetch model:
  - Only http/https URLs with a host are fetched (basic validation).
  - The SSRF guard (blocking localhost / private / link-local IPs) was removed at
    the project owner's request, so any reachable address may be fetched. Re-add
    IP/DNS checks in `is_safe_url` if deploying for untrusted users.
  - Responses are capped in size and time; downloaded bytes are never executed.
  - Secrets (API keys) are passed via headers only and are never logged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger("external_search")

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "external_cache"

USER_AGENT = os.getenv(
    "EXTERNAL_USER_AGENT",
    "AudioResearchAssistant/1.0 (research assistant; respects robots & rate limits)",
)
DEFAULT_TIMEOUT = float(os.getenv("EXTERNAL_HTTP_TIMEOUT", "12"))
MAX_BYTES = int(os.getenv("EXTERNAL_MAX_BYTES", str(3_000_000)))   # 3 MB hard cap
MAX_RETRIES = int(os.getenv("EXTERNAL_MAX_RETRIES", "2"))
# Short by default so freshly published papers/repos/pages appear within the hour
# (raise EXTERNAL_CACHE_TTL to trade freshness for fewer network calls).
CACHE_TTL = int(os.getenv("EXTERNAL_CACHE_TTL", str(60 * 60)))  # 1h

VALID_SOURCE_TYPES = ("local_pdf", "web", "github_repo", "github_code",
                      "online_pdf", "research_paper", "patent")


# ----------------------------------------------------------------------
# Source data structure
# ----------------------------------------------------------------------
@dataclasses.dataclass
class ExternalSource:
    """One piece of evidence (local or external). `text` is the content used for
    ranking + the LLM context; the rest is citation metadata."""
    source_type: str
    title: str
    url: str = ""
    snippet: str = ""
    text: str = ""
    provider: str = ""
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    page: Optional[int] = None
    published: Optional[str] = None
    license: Optional[str] = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"invalid source_type {self.source_type!r}")

    def content_hash(self) -> str:
        """Stable hash for de-duplication (url + path + a prefix of the text)."""
        basis = "|".join([
            (self.url or "").strip().lower().rstrip("/"),
            (self.file_path or "").strip().lower(),
            (self.text or self.snippet or "")[:400].strip().lower(),
        ])
        return hashlib.sha256(basis.encode("utf-8", "ignore")).hexdigest()[:16]

    def citation(self) -> str:
        """A short human-readable citation."""
        bits = [self.url or self.title]
        if self.file_path:
            loc = self.file_path
            if self.line_start:
                loc += f":{self.line_start}" + (f"-{self.line_end}" if self.line_end else "")
            bits.append(loc)
        if self.page:
            bits.append(f"p.{self.page}")
        return " · ".join(b for b in bits if b)

    def to_public(self) -> Dict[str, Any]:
        """Trimmed dict for the UI / SSE payload (no secrets)."""
        return {
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "snippet": (self.snippet or self.text or "")[:600],
            "text": (self.text or self.snippet or "")[:600],
            "provider": self.provider,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "page": self.page,
            "published": self.published,
            "license": self.license,
            "score": round(float(self.score), 3),
        }


# ----------------------------------------------------------------------
# URL validation
# ----------------------------------------------------------------------
# NOTE: The SSRF guard (blocking localhost / private / link-local / reserved
# addresses and re-resolving DNS) was removed at the project owner's request so
# external search can fetch any address. Only basic scheme/host validation
# remains. Re-add IP/DNS checks here if this is ever deployed for untrusted users.
def is_safe_url(url: str) -> Tuple[bool, str]:
    """Basic URL validation: require an http(s) URL with a host. No SSRF/private-IP
    blocking — every reachable address is allowed."""
    if not url or not isinstance(url, str):
        return False, "empty url"
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False, "unparseable url"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme {parsed.scheme!r} not allowed"
    if not parsed.hostname:
        return False, "missing host"
    return True, "ok"


# ----------------------------------------------------------------------
# TTL disk cache
# ----------------------------------------------------------------------
def _cache_file(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8", "ignore")).hexdigest()[:24]
    return CACHE_DIR / f"{digest}.json"


def cache_get(key: str, ttl: int = CACHE_TTL) -> Optional[Any]:
    path = _cache_file(key)
    if not path.exists():
        return None
    try:
        if (time.time() - path.stat().st_mtime) > ttl:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_set(key: str, value: Any) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_file(key).write_text(json.dumps(value), encoding="utf-8")
    except Exception as exc:  # caching is best-effort, never fatal
        logger.debug("cache write failed: %s", exc)


def cached(key: str, producer: Callable[[], Any], ttl: int = CACHE_TTL) -> Any:
    """Return cached value for `key`, else call producer(), cache, and return it."""
    hit = cache_get(key, ttl)
    if hit is not None:
        return hit
    value = producer()
    if value is not None:
        cache_set(key, value)
    return value


# ----------------------------------------------------------------------
# Guarded HTTP
# ----------------------------------------------------------------------
def safe_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
    expect: str = "text",          # "text" | "bytes" | "json"
    retries: int = MAX_RETRIES,
    data: Optional[Dict[str, Any]] = None,        # if set -> POST (form-encoded)
    json_body: Optional[Dict[str, Any]] = None,   # if set -> POST (JSON body)
) -> Optional[Any]:
    """HTTP GET (or POST when `data`/`json_body` is given) with URL validation,
    timeout, size cap, and retry-with-backoff on 429/5xx. Returns the text / bytes /
    parsed-json body, or None on any failure (never raises). Routing a JSON POST
    through here (instead of a raw requests.post) gives it the same rate-limit
    backoff the rest of external search already gets."""
    ok, reason = is_safe_url(url)
    if not ok:
        logger.warning("skipped invalid url (%s)", reason)
        return None

    hdrs = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    if headers:
        hdrs.update(headers)
    method = "POST" if (data is not None or json_body is not None) else "GET"

    for attempt in range(retries + 1):
        try:
            with requests.request(method, url, headers=hdrs, params=params, data=data,
                                  json=json_body, timeout=timeout, stream=True,
                                  allow_redirects=True) as resp:
                # Re-validate the final URL after redirects (scheme/host sanity).
                ok, reason = is_safe_url(resp.url)
                if not ok:
                    logger.warning("blocked redirect target (%s)", reason)
                    return None
                if resp.status_code >= 400:
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    return None
                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_content(8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        break
                    chunks.append(chunk)
                body = b"".join(chunks)[:max_bytes]
                if expect == "bytes":
                    return body
                if expect == "json":
                    try:
                        return json.loads(body.decode("utf-8", "ignore"))
                    except Exception:
                        return None
                encoding = resp.encoding or "utf-8"
                return body.decode(encoding, "ignore")
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            logger.info("fetch failed for host %s: %s", urlparse(url).hostname, type(exc).__name__)
            return None
    return None


def env_flag(name: str, default: bool = False) -> bool:
    return (os.getenv(name, "true" if default else "false") or "").strip().lower() in ("1", "true", "yes", "on")


# Generic filler/instruction words removed when turning a natural-language question
# into a keyword query for the search APIs. NOT a domain dictionary — it never adds
# or rewrites terms, only drops generic words, so retrieval stays broad.
_QUERY_STOP = {
    "explain", "describe", "give", "show", "write", "provide", "want", "need", "please",
    "tell", "implement", "implementation", "simulate", "simulation", "code", "program",
    "script", "runnable", "example", "examples", "demo", "using", "use", "get", "make",
    "build", "create", "compare", "find", "search", "python", "matlab", "java",
    "javascript", "the", "a", "an", "of", "for", "and", "or", "is", "are", "was", "were",
    "be", "to", "in", "on", "with", "how", "what", "why", "which", "does", "do", "can",
    "could", "would", "this", "that", "these", "those", "it", "its", "me", "my", "i",
    "you", "your", "we", "our", "as", "at", "by", "from", "about", "into", "then",
    "through", "like", "also", "vs", "versus", "please",
}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\+]*")


def clean_query(question: str, max_len: int = 200) -> str:
    """Turn a question into a compact keyword query for search APIs (the full
    question is still used for the LLM and for re-ranking). Falls back to the
    original if cleaning would empty it."""
    words = _WORD.findall((question or "").lower())
    kept = [w for w in words if w not in _QUERY_STOP and len(w) > 1]
    return (" ".join(kept).strip() or (question or "").strip())[:max_len]
