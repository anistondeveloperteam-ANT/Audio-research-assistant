"""
Event bus for distributed agent runs (optional, fallback-safe).

When the queue is on, a Celery worker publishes the agent's NDJSON events to a Redis list
`agent:events:{task_id}`; the web app's `/api/agent/{task_id}/stream` drains that list. Using
a LIST (not pub/sub) means a late subscriber still receives every buffered event.

Everything degrades to None/no-op when Redis is unreachable or `QUEUE_ENABLED` is off, so the
app falls back to in-process streaming and never breaks without Redis.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional

DONE_EVENT = {"__done__": True}
_TTL_SECONDS = 3600


def queue_enabled() -> bool:
    return os.getenv("QUEUE_ENABLED", "false").strip().lower() == "true"


def connect_redis() -> Optional[Any]:
    """Connect to REDIS_URL (no flag check) — used by the worker, which always needs Redis.
    Returns a live client or None."""
    try:
        import redis
        client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=2, socket_timeout=5)
        client.ping()
        return client
    except Exception:
        return None


def get_redis() -> Optional[Any]:
    """Flag-gated connection — used by the server to decide enqueue vs. in-process streaming."""
    if not queue_enabled():
        return None
    return connect_redis()


def _key(task_id: str) -> str:
    return f"agent:events:{task_id}"


def make_publisher(client: Any, task_id: str) -> Callable[[Dict[str, Any]], None]:
    """Return an on_event(event) callback that buffers events in Redis for the stream."""
    key = _key(task_id)

    def publish(event: Dict[str, Any]) -> None:
        try:
            client.rpush(key, json.dumps(event))
            client.expire(key, _TTL_SECONDS)
        except Exception:
            pass

    return publish


def finish(client: Any, task_id: str) -> None:
    """Mark the run complete so the stream consumer stops."""
    try:
        client.rpush(_key(task_id), json.dumps(DONE_EVENT))
        client.expire(_key(task_id), _TTL_SECONDS)
    except Exception:
        pass


def stream_events(client: Any, task_id: str, idle_limit: float = 300.0) -> Iterator[Dict[str, Any]]:
    """Yield buffered events for task_id until the DONE sentinel (or idle_limit of silence,
    e.g. a dead worker)."""
    key = _key(task_id)
    last = time.time()
    while True:
        try:
            item = client.blpop(key, timeout=2)
        except Exception:
            break
        if item is None:
            if time.time() - last > idle_limit:
                break
            continue
        last = time.time()
        try:
            event = json.loads(item[1])
        except Exception:
            continue
        if event.get("__done__"):
            break
        yield event
