"""
GitHub search via the public REST API (no scraping).

Finds relevant public repositories, reads their README / license / a few source
files, and returns them as `ExternalSource` records with repo URL + file path
(+ line range for code). Used only to *understand* algorithms/APIs — never to
copy code verbatim; the assistant reimplements ideas in this project's style and
cites the source + any license constraints.

Strict limits (env-tunable) keep API usage and payloads small:
    GITHUB_MAX_REPOS      (default 3)
    GITHUB_MAX_FILES      (default 3)   # code files per query (needs a token)
    GITHUB_MAX_FILE_BYTES (default 60000)
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

from backend.external_search.base import ExternalSource, cached, logger, safe_get

API = "https://api.github.com"
MAX_REPOS = int(os.getenv("GITHUB_MAX_REPOS", "5"))
MAX_FILES = int(os.getenv("GITHUB_MAX_FILES", "3"))
MAX_FILE_BYTES = int(os.getenv("GITHUB_MAX_FILE_BYTES", "60000"))
EXCERPT_LINES = int(os.getenv("GITHUB_EXCERPT_LINES", "120"))


def _headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"   # header only; never logged
    return h


def _api_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    return safe_get(url, headers=_headers(), params=params, expect="json")


def _decode_content(node: Dict[str, Any]) -> str:
    """Decode a GitHub contents/readme node (base64) to text, size-capped."""
    if not node:
        return ""
    if node.get("encoding") == "base64" and node.get("content"):
        try:
            raw = base64.b64decode(node["content"])[:MAX_FILE_BYTES]
            return raw.decode("utf-8", "ignore")
        except Exception:
            return ""
    return ""


def _excerpt(text: str) -> str:
    """Keep a leading excerpt so we cite, not dump, the file."""
    lines = text.splitlines()
    return "\n".join(lines[:EXCERPT_LINES])


# Default to most-starred so famous/popular repos surface; the source ranker still
# boosts recency when the query asks for the "latest". Override with
# GITHUB_REPO_SORT=updated for pure recency.
_REPO_SORT = os.getenv("GITHUB_REPO_SORT", "stars")


def search_repositories(query: str, max_repos: int = MAX_REPOS) -> List[Dict[str, Any]]:
    data = cached(
        f"gh_repos::{query}::{max_repos}::{_REPO_SORT}",
        lambda: _api_get(f"{API}/search/repositories",
                         {"q": query, "sort": _REPO_SORT, "order": "desc", "per_page": max_repos}),
    )
    if not data:
        return []
    return (data.get("items") or [])[:max_repos]


def fetch_readme(full_name: str) -> Optional[ExternalSource]:
    node = cached(f"gh_readme::{full_name}", lambda: _api_get(f"{API}/repos/{full_name}/readme"))
    if not node:
        return None
    text = _decode_content(node)
    if not text.strip():
        return None
    return ExternalSource(
        source_type="github_repo",
        title=f"{full_name} — README",
        url=node.get("html_url") or f"https://github.com/{full_name}",
        file_path=node.get("path") or "README.md",
        line_start=1,
        line_end=min(EXCERPT_LINES, len(text.splitlines())),
        text=_excerpt(text)[:8000],
        snippet=text.strip()[:600],
        provider="github",
    )


def fetch_license_name(full_name: str) -> Optional[str]:
    node = cached(f"gh_license::{full_name}", lambda: _api_get(f"{API}/repos/{full_name}/license"))
    if not node:
        return None
    lic = node.get("license") or {}
    return lic.get("spdx_id") or lic.get("name")


def search_code(query: str, max_files: int = MAX_FILES) -> List[ExternalSource]:
    """Code search (requires a GITHUB_TOKEN). Returns short, cited excerpts."""
    if not os.getenv("GITHUB_TOKEN"):
        return []
    data = cached(
        f"gh_code::{query}::{max_files}",
        lambda: _api_get(f"{API}/search/code", {"q": query, "per_page": max_files}),
    )
    if not data:
        return []
    out: List[ExternalSource] = []
    for item in (data.get("items") or [])[:max_files]:
        repo = (item.get("repository") or {}).get("full_name", "")
        path = item.get("path", "")
        node = cached(f"gh_file::{repo}::{path}",
                      lambda r=repo, p=path: _api_get(f"{API}/repos/{r}/contents/{p}"))
        text = _decode_content(node) if node else ""
        if not text.strip():
            continue
        out.append(ExternalSource(
            source_type="github_code",
            title=f"{repo} — {path}",
            url=item.get("html_url") or (node or {}).get("html_url") or "",
            file_path=path,
            line_start=1,
            line_end=min(EXCERPT_LINES, len(text.splitlines())),
            text=_excerpt(text)[:8000],
            snippet=text.strip()[:600],
            provider="github",
            license=fetch_license_name(repo),
        ))
    return out


def github_search(query: str, max_repos: int = MAX_REPOS) -> List[ExternalSource]:
    """Top repos' READMEs (+ code excerpts when a token is configured)."""
    sources: List[ExternalSource] = []
    for repo in search_repositories(query, max_repos=max_repos):
        full_name = repo.get("full_name")
        if not full_name:
            continue
        readme = fetch_readme(full_name)
        if readme:
            readme.license = (repo.get("license") or {}).get("spdx_id") or fetch_license_name(full_name)
            # Surface recency: when the repo was last pushed/updated.
            readme.published = (repo.get("pushed_at") or repo.get("updated_at") or "")[:10] or None
            # Surface popularity (★ stars) so famous repos are visible + citable.
            stars = repo.get("stargazers_count")
            if isinstance(stars, int):
                readme.title = f"{full_name} — README ({stars:,}★)"
                readme.snippet = f"★ {stars:,} stars · " + (readme.snippet or "")
            sources.append(readme)
    try:
        sources.extend(search_code(query))
    except Exception as exc:
        logger.info("github code search skipped: %s", type(exc).__name__)
    return sources
