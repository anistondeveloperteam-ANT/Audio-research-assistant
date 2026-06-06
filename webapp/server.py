"""
FastAPI server for the Audio Research Assistant web UI.

Run from the project root so `import backend.*` resolves:
    python run.py --web                 # -> http://localhost:8600
    uvicorn webapp.server:app --port 8600
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import chat_logic, ingest, settings
from backend.llm.streaming_provider import get_provider

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Audio Research Assistant")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ----------------------------------------------------------------------
# Page + config
# ----------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/config")
def config():
    try:
        prov = get_provider()
        provider_label = f"{prov.name} · {prov.model}"
    except Exception:
        provider_label = "unknown"
    # Web search is the primary source; local PDF RAG is optional/off by default.
    try:
        from backend.external_search import is_web_search_enabled
        web_search_available = is_web_search_enabled()
    except Exception:
        web_search_available = False
    return {
        "provider": provider_label,
        "web_search_available": web_search_available,
        "local_rag_enabled": chat_logic.ENABLE_LOCAL_RAG,
    }


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions():
    return chat_logic.memory().list_sessions(limit=50)


@app.post("/api/sessions")
def create_session():
    mem = chat_logic.memory()
    sid = mem.create_session()
    return mem.get_session(sid)


@app.put("/api/sessions/{session_id}")
def rename_session(session_id: str, body: dict = Body(default={})):
    title = (body.get("title") or "").strip() or "Untitled"
    chat_logic.memory().rename_session(session_id, title)
    return {"ok": True, "title": title}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    chat_logic.memory().delete_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/turns")
def get_turns(session_id: str):
    return chat_logic.memory().get_turns(session_id)


@app.delete("/api/sessions/{session_id}/turns/{turn_index}")
def delete_turn(session_id: str, turn_index: int):
    """Delete one question and its answer (a single user turn + the assistant
    reply that follows it)."""
    deleted = chat_logic.memory().delete_turn_pair(session_id, turn_index)
    return {"ok": True, "deleted": deleted}


@app.post("/api/sessions/{session_id}/turns/{turn_index}/truncate")
def truncate_turns(session_id: str, turn_index: int):
    """Delete the turn at turn_index and everything after it (used when the user
    edits an earlier question and we re-generate from that point)."""
    deleted = chat_logic.memory().delete_turns_from(session_id, turn_index)
    return {"ok": True, "deleted": deleted}


# ----------------------------------------------------------------------
# Library: upload a PDF + stream ingestion
# ----------------------------------------------------------------------
@app.get("/api/library")
def library():
    return ingest.library_stats()


@app.get("/api/papers")
def papers():
    return ingest.list_papers()


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: int):
    try:
        return ingest.delete_paper(paper_id)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ----------------------------------------------------------------------
# LLM model selection
# ----------------------------------------------------------------------
@app.get("/api/models")
def models():
    return settings.list_models()


@app.post("/api/model")
def set_model(body: dict = Body(...)):
    try:
        return settings.set_model(body.get("provider", ""), body.get("model", ""))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    return ingest.save_pdf(file.filename or "paper.pdf", data)


@app.post("/api/ingest")
def run_ingest():
    def gen():
        try:
            for event in ingest.stream_ingest():
                yield json.dumps(event) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ----------------------------------------------------------------------
# Chat (streaming, newline-delimited JSON)
# ----------------------------------------------------------------------
@app.post("/api/chat")
def chat(body: dict = Body(...)):
    session_id = body.get("session_id")
    question = body.get("question", "")
    mode = body.get("mode", "Default")
    top_k = body.get("top_k", 8)
    web_search = bool(body.get("web_search", True))   # web search is the default source
    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)

    def gen():
        try:
            for event in chat_logic.stream_chat_events(
                session_id, question, mode=mode, top_k=top_k, web_search=web_search
            ):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # last-resort guard so the stream always closes cleanly
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    # Sync generator -> Starlette iterates it in a threadpool (safe for blocking calls).
    return StreamingResponse(gen(), media_type="application/x-ndjson")
