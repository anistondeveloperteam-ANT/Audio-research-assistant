"""Optional, fully self-contained LLM observability (Langfuse tracing).

Disabled by default. With LANGFUSE_ENABLED unset/false the public helpers return
no-op singletons, so importing and calling them is byte-identical to not tracing
at all — no client, no network, no `langfuse` import. See backend/observability/tracing.py.
"""
from backend.observability.tracing import flush, start_trace

__all__ = ["start_trace", "flush"]
