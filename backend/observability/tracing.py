"""
Langfuse tracing — optional, safe, and zero-overhead when off.

Design contract (see docs/OBSERVABILITY.md):
  * Tracing is active ONLY when LANGFUSE_ENABLED=true AND LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are all set. Env *names* are read
    here; values are never logged and the real .env is never read by this module.
  * When disabled (the default) every public call returns a shared no-op object.
    No client is built, `langfuse` is never imported, nothing is timed or sent —
    behaviour is byte-identical to not calling tracing at all.
  * Every call is wrapped in try/except: a missing package, an unreachable server,
    or any SDK error degrades to a no-op and NEVER raises into the chat flow.
  * Spans carry only durations + small metadata (counts, scores, booleans, short
    summaries). Callers must never pass full user text, source bodies, or secrets.

Usage:
    trace = start_trace("chat_request", mode=mode)
    with trace.span("local_rag") as sp:
        items = retrieve(...)
        sp.set(count=len(items))
    trace.end()
"""
from __future__ import annotations

import atexit
import os
from typing import Any, Optional

# How much of any free-text summary may reach a span (privacy guard).
_SUMMARY_CAP = 500

_client: Any = None
_client_tried = False


def _enabled() -> bool:
    """True only when explicitly turned on AND fully configured. Cheap (env only);
    evaluated per request so live env / tests are respected (not baked at import)."""
    return (
        os.getenv("LANGFUSE_ENABLED", "false").strip().lower() == "true"
        and bool(os.getenv("LANGFUSE_HOST"))
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def _get_client() -> Any:
    """Lazily build the Langfuse client (it reads LANGFUSE_* from env itself).
    Returns None on any failure and never retries a failed init."""
    global _client, _client_tried
    if _client is not None:
        return _client
    if _client_tried:
        return None
    _client_tried = True
    try:
        from langfuse import Langfuse  # imported only when actually enabled
        _client = Langfuse()
        atexit.register(flush)  # best-effort send of buffered spans at shutdown
        return _client
    except Exception:
        return None


def _clean(meta: Optional[dict]) -> Optional[dict]:
    """Drop None values and cap any string to _SUMMARY_CAP chars."""
    if not meta:
        return None
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v[:_SUMMARY_CAP] if isinstance(v, str) else v
    return out or None


class _Span:
    """Wraps a Langfuse span as a context manager: records metadata + success/
    failure on exit, then ends the span. Never suppresses exceptions."""

    __slots__ = ("_span", "_meta")

    def __init__(self, span: Any):
        self._span = span
        self._meta: dict = {}

    def set(self, **meta: Any) -> "_Span":
        self._meta.update(meta)
        return self

    def __enter__(self) -> "_Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is not None:
                self._span.update(level="ERROR",
                                  status_message=str(exc)[:_SUMMARY_CAP],
                                  metadata=_clean({**self._meta, "ok": False}))
            else:
                self._span.update(metadata=_clean(self._meta))
            self._span.end()
        except Exception:
            pass
        return False  # never swallow the caller's exception


class _Trace:
    """Wraps the root Langfuse span; `.span()` opens child spans (explicit parent,
    so they nest correctly even when created on a worker thread)."""

    __slots__ = ("_root",)

    def __init__(self, root: Any):
        self._root = root

    def span(self, name: str, **meta: Any) -> Any:
        try:
            child = self._root.start_observation(
                name=name, as_type="span", metadata=_clean(meta))
            return _Span(child)
        except Exception:
            return _NOOP_SPAN

    def set(self, **meta: Any) -> "_Trace":
        try:
            self._root.update(metadata=_clean(meta))
        except Exception:
            pass
        return self

    def end(self) -> None:
        try:
            self._root.end()
        except Exception:
            pass


class _NoopSpan:
    """Shared do-nothing span used whenever tracing is off."""

    __slots__ = ()

    def set(self, **meta: Any) -> "_NoopSpan":
        return self

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _NoopTrace:
    """Shared do-nothing trace used whenever tracing is off."""

    __slots__ = ()

    def span(self, name: str, **meta: Any) -> "_NoopSpan":
        return _NOOP_SPAN

    def set(self, **meta: Any) -> "_NoopTrace":
        return self

    def end(self) -> None:
        return None


_NOOP_SPAN = _NoopSpan()
_NOOP_TRACE = _NoopTrace()


def start_trace(name: str, **meta: Any) -> Any:
    """Begin one trace for a request. Returns a no-op trace when disabled or on
    any error, so callers can always use `with trace.span(...)` unconditionally."""
    if not _enabled():
        return _NOOP_TRACE
    client = _get_client()
    if client is None:
        return _NOOP_TRACE
    try:
        root = client.start_observation(name=name, as_type="span", metadata=_clean(meta))
        return _Trace(root)
    except Exception:
        return _NOOP_TRACE


def flush() -> None:
    """Best-effort flush of buffered spans (no-op when disabled). Non-blocking on
    the hot path: callers end spans instead; this is only used at shutdown."""
    if not _enabled():
        return
    try:
        client = _get_client()
        if client is not None:
            client.flush()
    except Exception:
        pass
