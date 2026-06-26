"""
request_context.py — per-request, concurrency-safe settings (no process-global mutation).

Per-request configuration (Fast/Deep run profile, the selected chat model, and every derived knob) must
NOT live in process-global ``os.environ``: under concurrency one request would overwrite another's
settings mid-flight and answers would be computed with the wrong config. Instead each request resolves
its settings ONCE and binds them to a ``contextvars.ContextVar``; the typed readers below return that
request's value, falling back to ``os.environ`` (startup/static config) then a default when no request
is active — so the CLI, the eval harnesses, and the test suite keep working unchanged.

Two concurrency realities are handled explicitly:

* **Streaming generators.** Starlette iterates a sync streaming response one chunk at a time, each in a
  freshly *copied* context, so a ``ContextVar.set()`` inside the generator would not survive across
  ``yield``s. ``bind_to_context()`` snapshots the context once and runs every ``next()`` inside it, so a
  request's settings persist for the whole streamed answer.
* **Thread pools.** The pipeline fans out work on ``ThreadPoolExecutor``s whose worker threads start with
  an empty context. ``ContextThreadPoolExecutor`` copies the submitting thread's context into each
  worker, so retrieval/external/agent workers read the same per-request settings.
"""
from __future__ import annotations

import concurrent.futures
import contextvars
import functools
import os
from typing import Any, Dict, Iterator, Optional

# The single per-request settings map. None = no request bound (CLI / tests / startup) -> env fallback.
_settings: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "request_settings", default=None)

_TRUE = {"1", "true", "yes", "on"}


# ----------------------------------------------------------------------
# Bind / read the per-request settings
# ----------------------------------------------------------------------
def set_request_settings(settings: Dict[str, Any]) -> "contextvars.Token":
    """Bind a resolved settings map to the CURRENT context. Returns a token for `reset_request_settings`.
    Stores a shallow copy so later mutation of the caller's dict can't bleed across requests."""
    return _settings.set(dict(settings or {}))


def reset_request_settings(token: "contextvars.Token") -> None:
    try:
        _settings.reset(token)
    except (ValueError, LookupError):                  # token from another context / already reset
        pass


def clear_request_settings() -> None:
    """Drop any bound settings (back to env-fallback). For tests and between-request cleanup."""
    _settings.set(None)


def has_request_settings() -> bool:
    """True when a request has bound its settings (so callers can avoid clobbering an outer binding)."""
    return _settings.get() is not None


def current_settings() -> Dict[str, Any]:
    return dict(_settings.get() or {})


def _ctx_value(name: str):
    s = _settings.get()
    if s is not None and name in s and s[name] is not None:
        return s[name]
    return None


def request_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = _ctx_value(name)
    if v is not None:
        return str(v)
    env = os.getenv(name)
    return env if env is not None else default


def request_int(name: str, default: int) -> int:
    v = _ctx_value(name)
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return default
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def request_float(name: str, default: float) -> float:
    v = _ctx_value(name)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def request_bool(name: str, default: bool) -> bool:
    v = _ctx_value(name)
    if v is not None:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in _TRUE
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


# ----------------------------------------------------------------------
# Concurrency propagation
# ----------------------------------------------------------------------
def _context_bound_iterator(ctx: "contextvars.Context", iterator: Iterator) -> Iterator:
    class _Bound:
        def __iter__(self):
            return self

        def __next__(self):
            return ctx.run(next, iterator)

    return _Bound()


def bind_to_context(iterator: Iterator) -> Iterator:
    """Wrap a generator so EVERY ``next()`` runs inside one snapshot of the CURRENT context (the caller
    must have already bound settings). Keeps settings alive across the streamed ``yield``s even though
    Starlette copies a fresh context per chunk."""
    return _context_bound_iterator(contextvars.copy_context(), iterator)


def bound_stream(settings: Dict[str, Any], iterator: Iterator) -> Iterator:
    """For a streaming endpoint: run every ``next()`` of ``iterator`` inside a FRESH context that has
    ``settings`` bound — WITHOUT mutating the caller's context. This is the robust streaming entry: the
    (pooled) endpoint thread is never polluted, so the isolation does not depend on the server copying
    the context per call."""
    ctx = contextvars.copy_context()
    ctx.run(set_request_settings, settings)
    return _context_bound_iterator(ctx, iterator)


def run_with_settings(settings: Dict[str, Any], target):
    """Return a 0-arg callable that runs ``target()`` inside a FRESH context with ``settings`` bound —
    for a raw ``threading.Thread`` worker at a REQUEST ENTRY (the agent / research runs). The caller's
    context is untouched, so the worker reads exactly this request's settings and nothing leaks back
    onto the spawning thread."""
    ctx = contextvars.copy_context()
    ctx.run(set_request_settings, settings)
    return functools.partial(ctx.run, target)


def run_in_current_context(target):
    """Return a 0-arg callable that runs ``target()`` inside a copy of the CURRENT context — for a raw
    ``threading.Thread`` worker spawned MID-PIPELINE (settings already bound), e.g. the code agent
    launched from inside the chat stream. The worker inherits whatever the spawning thread has bound."""
    ctx = contextvars.copy_context()
    return functools.partial(ctx.run, target)


class ContextThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` that runs every submitted task inside a COPY of the submitting thread's
    context, so worker threads read the same per-request settings. Drop-in replacement at the executor
    creation site; ``map`` inherits this via ``submit``."""

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, functools.partial(fn, *args, **kwargs))
