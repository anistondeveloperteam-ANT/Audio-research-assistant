"""
config.py — data paths + Oracle credentials, loaded from .env.

Only the values that are actually imported elsewhere live here: PAPERS_DIR and
the ORACLE_* credentials (used by pipeline.py and webapp/ingest.py). Embedding,
retrieval, and chat-model settings are read directly from the environment by the
modules that use them (backend/common/embeddings.py, backend/retrieval/*,
backend/llm/streaming_provider.py).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACTED_DIR = DATA_DIR / "extracted"

for folder in (DATA_DIR, PAPERS_DIR, EXTRACTED_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ORACLE_USER = os.getenv("ORACLE_USER", "AUDIO_RAG")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "change_me")
ORACLE_DSN = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")
