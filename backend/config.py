import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACTED_DIR = DATA_DIR / "extracted"
LOG_DIR = DATA_DIR / "logs"

for folder in [DATA_DIR, PAPERS_DIR, EXTRACTED_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ORACLE_USER = os.getenv("ORACLE_USER", "AUDIO_RAG")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "AudioRagPass123")
ORACLE_DSN = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

MAX_QUERY_ROUTES = int(os.getenv("MAX_QUERY_ROUTES", "5"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
TOTAL_SOURCE_LIMIT = int(os.getenv("TOTAL_SOURCE_LIMIT", "16"))
PER_TOPIC_SOURCE_LIMIT = int(os.getenv("PER_TOPIC_SOURCE_LIMIT", "3"))

ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "manual").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_DEEP_MODEL = os.getenv("ANTHROPIC_DEEP_MODEL", ANTHROPIC_MODEL)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_DEEP_MODEL = os.getenv("OPENAI_DEEP_MODEL", OPENAI_MODEL)

MANUAL_MODE = ANSWER_PROVIDER == "manual"