"""Shared pytest setup: make the project root importable, and keep the suite fully offline."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The offline suite must NEVER load a real torch model. The external-search reranker otherwise loads
# a real CrossEncoder (BAAI/bge-reranker-v2-m3) whenever the dev's .env has ENABLE_LOCAL_RAG=true —
# slow, and it intermittently segfaults transformers/torch on Windows (a native access violation).
# Force the lexical scorer here (rerank_sources already supports that fallback). Set before any
# backend module imports source_ranker so its module-level USE_CROSS_ENCODER reads this value.
os.environ["EXTERNAL_RERANK_CROSS_ENCODER"] = "false"
