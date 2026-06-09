"""
FastAPI server for the Research Assistant web UI.

Run from the project root so `import backend.*` resolves:
    python run.py --web                 # -> http://localhost:8600
    uvicorn webapp.server:app --port 8600

Multi-user (optional): set ENABLE_AUTH=true and create accounts with
    python -m backend.auth.users add <user_id>
Members then sign in at /login; each member's conversations are private.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse, JSONResponse, RedirectResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import auth as webauth, chat_logic, ingest, settings
from backend.auth.users import create_user, verify_user
from backend.llm.streaming_provider import get_provider

STATIC = Path(__file__).resolve().parent / "static"


def require_login(request: Request) -> None:
    """Global gate: when ENABLE_AUTH, every non-public route needs a session."""
    if not webauth.auth_enabled() or webauth.is_public_path(request.url.path):
        return
    if not request.session.get("user_id"):
        raise HTTPException(status_code=401, detail="Authentication required")


app = FastAPI(title="Research Assistant", dependencies=[Depends(require_login)])
# Signs the session cookie. By default the cookie is a *session* cookie (max_age=None):
# it is cleared when the browser closes, so each new visit requires signing in again.
# Set SESSION_MAX_AGE=<seconds> in .env to keep users logged in for that long instead.
app.add_middleware(
    SessionMiddleware, secret_key=webauth.session_secret(),
    same_site="lax", https_only=False, max_age=webauth.session_max_age(),
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _require_owner(request: Request, session_id: str) -> None:
    """Ensure the current user owns this conversation (404 if missing, 403 if not theirs)."""
    owner = chat_logic.memory().session_owner(session_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="No such conversation")
    if owner != webauth.current_user(request):
        raise HTTPException(status_code=403, detail="Not your conversation")


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
_NO_STORE = {"Cache-Control": "no-store, must-revalidate"}


@app.get("/")
def index(request: Request):
    if webauth.auth_enabled() and not request.session.get("user_id"):
        return RedirectResponse("/login")
    # no-store so the browser always re-checks auth on "/" (never serves a stale shell).
    return FileResponse(str(STATIC / "index.html"), headers=_NO_STORE)


@app.get("/login")
def login_page():
    return FileResponse(str(STATIC / "login.html"), headers=_NO_STORE)


# ----------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------
@app.get("/api/me")
def whoami(request: Request):
    if not webauth.auth_enabled():
        return {"auth": False, "user_id": webauth.LOCAL_USER}
    return {
        "auth": True,
        "user_id": request.session.get("user_id"),
        "signup": webauth.signup_enabled(),
    }


@app.post("/api/login")
def api_login(request: Request, body: dict = Body(default={})):
    uid = (body.get("user_id") or "").strip()
    pw = body.get("password") or ""
    if verify_user(uid, pw):
        request.session["user_id"] = uid
        return {"ok": True, "user_id": uid}
    return JSONResponse({"ok": False, "error": "Invalid user ID or password."},
                        status_code=401)


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.post("/api/signup")
def api_signup(request: Request, body: dict = Body(default={})):
    if not webauth.signup_enabled():
        return JSONResponse(
            {"error": "Sign-ups are disabled. Ask an admin to create your account."},
            status_code=403)
    uid = (body.get("user_id") or "").strip()
    pw = body.get("password") or ""
    try:
        create_user(uid, pw)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    request.session["user_id"] = uid
    return {"ok": True, "user_id": uid}


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
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
        "local_rag_enabled": chat_logic.local_rag_enabled(),
    }


# ----------------------------------------------------------------------
# Sessions (scoped to the current user)
# ----------------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions(request: Request):
    return chat_logic.memory().list_sessions(limit=50, user_id=webauth.current_user(request))


@app.post("/api/sessions")
def create_session(request: Request):
    mem = chat_logic.memory()
    sid = mem.create_session(user_id=webauth.current_user(request))
    return mem.get_session(sid)


@app.put("/api/sessions/{session_id}")
def rename_session(session_id: str, request: Request, body: dict = Body(default={})):
    _require_owner(request, session_id)
    title = (body.get("title") or "").strip() or "Untitled"
    chat_logic.memory().rename_session(session_id, title)
    return {"ok": True, "title": title}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    _require_owner(request, session_id)
    chat_logic.memory().delete_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/turns")
def get_turns(session_id: str, request: Request):
    _require_owner(request, session_id)
    return chat_logic.memory().get_turns(session_id)


@app.delete("/api/sessions/{session_id}/turns/{turn_index}")
def delete_turn(session_id: str, turn_index: int, request: Request):
    """Delete one question and its answer (a single user turn + the assistant
    reply that follows it)."""
    _require_owner(request, session_id)
    deleted = chat_logic.memory().delete_turn_pair(session_id, turn_index)
    return {"ok": True, "deleted": deleted}


@app.post("/api/sessions/{session_id}/turns/{turn_index}/truncate")
def truncate_turns(session_id: str, turn_index: int, request: Request):
    """Delete the turn at turn_index and everything after it (used when the user
    edits an earlier question and we re-generate from that point)."""
    _require_owner(request, session_id)
    deleted = chat_logic.memory().delete_turns_from(session_id, turn_index)
    return {"ok": True, "deleted": deleted}


# ----------------------------------------------------------------------
# Library: upload a PDF + stream ingestion (shared knowledge base)
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
def chat(request: Request, body: dict = Body(...)):
    session_id = body.get("session_id")
    question = body.get("question", "")
    mode = body.get("mode", "Default")
    top_k = body.get("top_k", 8)
    web_search = bool(body.get("web_search", True))   # web search is the default source
    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)
    _require_owner(request, session_id)   # raises before streaming if not the user's

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


# ----------------------------------------------------------------------
# Agent mode (write code -> run in Docker -> verify -> refine), streamed
# ----------------------------------------------------------------------
@app.post("/api/agent")
def agent(body: dict = Body(...)):
    task = (body.get("question") or body.get("task") or "").strip()
    use_search = bool(body.get("use_search", False))   # off by default = faster
    if not task:
        return JSONResponse({"error": "task is required"}, status_code=400)

    import queue
    import threading
    from backend.agent.loop import run_agent

    q: "queue.Queue" = queue.Queue()
    DONE = object()

    def worker():
        try:
            run_agent(task, use_search=use_search, on_event=q.put)
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            event = q.get()
            if event is DONE:
                break
            yield json.dumps(event) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ----------------------------------------------------------------------
# Deep research agent (plan -> search everywhere -> reflect -> report), streamed
# ----------------------------------------------------------------------
@app.post("/api/research")
def research(body: dict = Body(...)):
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    import queue
    import threading
    from backend.agent.research_agent import research as run_research

    q: "queue.Queue" = queue.Queue()
    DONE = object()

    def worker():
        try:
            res = run_research(question, on_event=q.put)
            q.put({"type": "result", "report": res.report, "sources": res.sources,
                   "sub_questions": res.sub_questions, "rounds": res.rounds,
                   "review": res.review, "error": res.error})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            event = q.get()
            if event is DONE:
                break
            yield json.dumps(event) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ----------------------------------------------------------------------
# Review: structured peer review of an answer or pasted text
# ----------------------------------------------------------------------
@app.post("/api/review")
def api_review(body: dict = Body(...)):
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    try:
        from backend.answering.reviewer import review
        return review(text)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
