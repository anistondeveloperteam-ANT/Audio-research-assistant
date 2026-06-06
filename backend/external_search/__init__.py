"""
External knowledge retrieval (optional, opt-in via ENABLE_WEB_SEARCH).

A small provider layer that lets the assistant search the public web, GitHub
repos/code, and online PDFs, read the pages safely, and feed the extracted text
back into the existing RAG pipeline as a *separate* evidence channel. The local
PDF retrieval is unchanged; this is purely additive.

Public entry point:
    from backend.external_search import gather_external_evidence
    sources, warnings = gather_external_evidence("your query", max_results=6)
"""
from backend.external_search.orchestrator import gather_external_evidence, is_web_search_enabled

__all__ = ["gather_external_evidence", "is_web_search_enabled"]
