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

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import chat_logic
from backend.answering.research_modes import MODE_SETTINGS
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
    return {
        "modes": list(MODE_SETTINGS.keys()),       # Fast / Balanced / Deep
        "default_mode": "Balanced",
        "default_top_k": 8,
        "provider": provider_label,
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


# ----------------------------------------------------------------------
# Chat (streaming, newline-delimited JSON)
# ----------------------------------------------------------------------
@app.post("/api/chat")
def chat(body: dict = Body(...)):
    session_id = body.get("session_id")
    question = body.get("question", "")
    mode = body.get("mode", "Balanced")
    top_k = body.get("top_k", 8)
    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)

    def gen():
        try:
            for event in chat_logic.stream_chat_events(session_id, question, mode=mode, top_k=top_k):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # last-resort guard so the stream always closes cleanly
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    # Sync generator -> Starlette iterates it in a threadpool (safe for blocking calls).
    return StreamingResponse(gen(), media_type="application/x-ndjson")
