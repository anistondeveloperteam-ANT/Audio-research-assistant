"""
Celery app for distributed agent execution (optional).

Broker + result backend: REDIS_URL (default redis://localhost:6379/0). Only used when
QUEUE_ENABLED=true; otherwise the web app runs the agent in-process and this is never touched.

Run a worker (Windows needs the solo pool):

    celery -A backend.agent.celery_app worker --pool=solo --loglevel=info
"""
from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery("audio_research_agent", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
)


@app.task(name="agent.run_agent_graph")
def run_agent_graph_task(task_id: str, task: str, brief: str = "", conversation: str = "") -> None:
    """Run the multi-agent graph in the worker, streaming events to Redis for the web app."""
    from backend.agent.graph import run_agent_graph
    from backend.agent.task_channel import connect_redis, finish, make_publisher

    client = connect_redis()
    publish = make_publisher(client, task_id) if client is not None else (lambda e: None)
    try:
        run_agent_graph(task, brief=brief, conversation=conversation,
                        task_id=task_id, on_event=publish)
    finally:
        if client is not None:
            finish(client, task_id)
