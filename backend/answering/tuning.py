"""
tuning.py — the self-tuning OVERRIDE LAYER (Phase 3: eval-gated config calibration).

Every numeric threshold getter in the pipeline (recall floors, CRAG grade cuts, answer-cache
similarity, ...) routes its computed value through `tuned(name, fallback)`. When the eval-gated tuner
has PROVEN a better value, that override is returned (clamped to the tunable's bounds); otherwise the
getter's own env/default `fallback` is returned unchanged. With no overrides set, this layer is a pure
no-op — stock behaviour — so merely shipping Phase 3 changes nothing until the owner runs the tuner.

Zero latency on the hot path: `tuned()` is a dict lookup against an in-process cache. The cache is
refreshed from the DB at most once per TTL via `refresh(mem)` (called once at the top of an answer),
never per threshold read. Overrides are bounded, persisted, and fully reversible (`clear_overrides`).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Tunable:
    """One calibratable knob: the env var it overrides, its default, sane [lo, hi] bounds, the search
    step the tuner moves by, and whether it is integer-valued."""
    name: str
    default: float
    lo: float
    hi: float
    step: float
    is_int: bool = False


def _t(name: str, default: float, lo: float, hi: float, step: float, is_int: bool = False) -> Tunable:
    return Tunable(name=name, default=default, lo=lo, hi=hi, step=step, is_int=is_int)


# The registry of tunables. Bounds are conservative — the tuner can only move WITHIN these, so a bad
# search can never push a threshold somewhere absurd. Keyed by the env-var name the getter reads.
TUNABLES: Dict[str, Tunable] = {
    "EXPERIENCE_MIN_RELEVANCE": _t("EXPERIENCE_MIN_RELEVANCE", 0.62, 0.40, 0.90, 0.03),
    "EXPERIENCE_TOP_K":         _t("EXPERIENCE_TOP_K", 3, 1, 8, 1, is_int=True),
    "EXPERIENCE_HALF_LIFE_DAYS": _t("EXPERIENCE_HALF_LIFE_DAYS", 30.0, 5.0, 180.0, 5.0),
    "CORPUS_MIN_RELEVANCE":     _t("CORPUS_MIN_RELEVANCE", 0.5, 0.35, 0.90, 0.03),
    "CORPUS_TOP_K":             _t("CORPUS_TOP_K", 3, 1, 8, 1, is_int=True),
    "CORPUS_HALF_LIFE_DAYS":    _t("CORPUS_HALF_LIFE_DAYS", 120.0, 15.0, 365.0, 15.0),
    "CRAG_STRONG_MIN":          _t("CRAG_STRONG_MIN", 0.55, 0.35, 0.80, 0.03),
    "CRAG_PARTIAL_MIN":         _t("CRAG_PARTIAL_MIN", 0.30, 0.10, 0.55, 0.03),
    "CRAG_STRONG_COUNT":        _t("CRAG_STRONG_COUNT", 2, 1, 5, 1, is_int=True),
    "ANSWER_CACHE_MIN_SIMILARITY": _t("ANSWER_CACHE_MIN_SIMILARITY", 0.97, 0.92, 1.0, 0.01),
    "ANSWER_CACHE_MIN_SEMANTIC":   _t("ANSWER_CACHE_MIN_SEMANTIC", 0.88, 0.80, 0.99, 0.01),
}


def tunable_names() -> List[str]:
    return list(TUNABLES.keys())


def clamp(t: Tunable, value: float) -> float:
    """Clamp `value` into the tunable's bounds, rounding to an int when it is integer-valued. A non-finite
    value (NaN/inf from a bad evaluator or a hand-edited DB row) falls back to the default — never
    propagates to a getter (where int(round(nan)) would raise)."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return t.default
    if not math.isfinite(fv):
        return t.default
    v = max(t.lo, min(t.hi, fv))
    return float(int(round(v))) if t.is_int else v


# ----------------------------------------------------------------------
# Zero-latency override cache. `_OVERRIDES` is the only thing the hot path touches.
# ----------------------------------------------------------------------
_lock = threading.Lock()
_OVERRIDES: Dict[str, float] = {}
_LOADED_AT: float = 0.0
_PINNED: bool = False                                  # when the tuner is evaluating a candidate config


def tuned(name: str, fallback: float) -> float:
    """The active override for `name` (clamped), or `fallback` (the getter's env/default) when there is
    none. Pure dict lookup — no DB, no env read, no latency. Unknown names pass through to fallback."""
    t = TUNABLES.get(name)
    if t is None:
        return fallback
    with _lock:
        v = _OVERRIDES.get(name)
    if v is None:
        return fallback
    return clamp(t, v)


def refresh(mem, *, ttl: float = 60.0, force: bool = False) -> None:
    """Reload the override cache from the DB at most once per `ttl` seconds (or immediately when
    `force`). Fail-open: a DB error leaves the current cache intact. Call once at the top of an answer —
    the per-threshold reads then stay zero-latency.

    Cross-process note: in-process `set_override`/`clear_overrides` force-refresh, so changes are instant
    within the process that made them. A DIFFERENT process (e.g. the offline tuner CLI clearing config)
    is picked up within `ttl` seconds (self-healing, bounded) — restart or wait out the TTL for an
    immediate effect after an out-of-band change."""
    global _OVERRIDES, _LOADED_AT
    now = time.time()
    with _lock:
        if _PINNED and not force:                      # the tuner has pinned a candidate; don't clobber it
            return
        fresh_enough = _LOADED_AT and (now - _LOADED_AT) < ttl
    if fresh_enough and not force:
        return
    try:
        loaded = mem.get_tuned_config()
    except Exception:                                  # noqa: BLE001 - tuning must never break a turn
        loaded = None
    with _lock:
        if loaded is not None:
            _OVERRIDES = {k: float(v) for k, v in loaded.items() if k in TUNABLES}
        _LOADED_AT = now


def current_overrides() -> Dict[str, float]:
    """A snapshot of the active overrides (clamped) — for display / the tuner's starting point."""
    with _lock:
        items = list(_OVERRIDES.items())
    return {k: clamp(TUNABLES[k], v) for k, v in items if k in TUNABLES}


def set_cache(overrides: Optional[Dict[str, float]]) -> None:
    """Replace the in-process override cache directly (no DB). Used by the tuner while evaluating a
    candidate, and by tests. Only known tunables are kept."""
    global _OVERRIDES, _LOADED_AT
    with _lock:
        _OVERRIDES = {k: float(v) for k, v in (overrides or {}).items() if k in TUNABLES}
        _LOADED_AT = time.time()


def clear_cache() -> None:
    """Drop all in-process overrides (revert to stock defaults). Does not touch the DB."""
    global _OVERRIDES, _LOADED_AT, _PINNED
    with _lock:
        _OVERRIDES = {}
        _LOADED_AT = 0.0
        _PINNED = False


def pin(overrides: Optional[Dict[str, float]]) -> None:
    """PIN a candidate config in the cache so the pipeline's `refresh(mem)` won't reload over it while
    the tuner is measuring that candidate. Use via the tuner's `pinned()` context manager.

    OFFLINE-ONLY CONTRACT: the cache + pin are PROCESS-GLOBAL, so the tuner must run as its own offline
    process (the CLI). Pinning inside a live server process would transiently expose the un-proven
    candidate to concurrent requests. `pin`/`tune` are deliberately referenced nowhere on the request
    path — keep it that way."""
    global _PINNED
    set_cache(overrides)
    with _lock:
        _PINNED = True


def unpin() -> None:
    """Release a pin and force the next `refresh(mem)` to reload the real persisted overrides."""
    global _PINNED, _LOADED_AT
    with _lock:
        _PINNED = False
        _LOADED_AT = 0.0


def set_override(mem, name: str, value: float, *, source: str = "self_tuner") -> Optional[float]:
    """Persist a bounded override and make it live immediately. Returns the clamped value stored, or
    None for an unknown tunable."""
    t = TUNABLES.get(name)
    if t is None:
        return None
    v = clamp(t, value)
    mem.set_tuned_config(name, v, source=source)
    refresh(mem, force=True)
    return v


def clear_overrides(mem, name: Optional[str] = None) -> int:
    """Revert one override (or all) in the DB and refresh the cache. Returns how many were removed."""
    removed = mem.clear_tuned_config(name)
    refresh(mem, force=True)
    return removed
