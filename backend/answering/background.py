"""
background.py — a tiny fire-and-forget dispatcher so LEARNING never adds latency to an answer.

The answer streams to the user immediately; slow post-answer work — embedding verified findings and
updating the grown RAG corpus (Phase 2) — runs on a worker thread AFTER the response is delivered.
Stdlib only (`concurrent.futures.ThreadPoolExecutor`); no new dependency. A failed background task is
swallowed + logged and can NEVER surface to the user. `flush()` lets tests/shutdown await pending work
for determinism, and `LEARN_BACKGROUND_SYNC=1` forces inline execution (tests/debug).
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, List

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_executor: "ThreadPoolExecutor | None" = None
_pending: List[Future] = []


def _max_workers() -> int:
    try:
        return max(1, int(os.getenv("LEARN_BACKGROUND_WORKERS", "2")))
    except (TypeError, ValueError):
        return 2


def _sync() -> bool:
    return os.getenv("LEARN_BACKGROUND_SYNC", "").strip().lower() in ("1", "true", "yes", "on")


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_max_workers(), thread_name_prefix="learn")
            atexit.register(lambda: _executor and _executor.shutdown(wait=False))
        return _executor


def _guard(fn: Callable, args: tuple, kwargs: dict) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:                                  # noqa: BLE001 - background learn never raises
        logger.warning("background learn task failed", exc_info=True)


def run(fn: Callable, *args, **kwargs) -> None:
    """Submit `fn(*args, **kwargs)` to run OFF the request path. Fail-open: a dispatch failure is logged
    and dropped rather than blocking the answer. Runs inline when LEARN_BACKGROUND_SYNC is set."""
    if _sync():
        _guard(fn, args, kwargs)
        return
    try:
        fut = _get_executor().submit(_guard, fn, args, kwargs)
        with _lock:
            # Drop already-finished futures every dispatch so the list can't grow unbounded across a
            # long run (we only ever need the still-pending ones, for flush()).
            _pending[:] = [f for f in _pending if not f.done()]
            _pending.append(fut)
    except Exception:                                  # noqa: BLE001 - never let scheduling break a turn
        logger.warning("could not dispatch background learn task", exc_info=True)


def flush(timeout: float = 10.0) -> None:
    """Block until currently-pending tasks finish (test/shutdown determinism). Never raises. Clears the
    finished futures it waited on so the pending list doesn't retain them."""
    with _lock:
        pending = [f for f in _pending if not f.done()]
    for f in pending:
        try:
            f.result(timeout=timeout)
        except Exception:                              # noqa: BLE001
            pass
    with _lock:
        _pending[:] = [f for f in _pending if not f.done()]
