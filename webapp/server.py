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
import os
import secrets
import sys
import time as _time
from collections import defaultdict, deque
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
from backend.auth.users import (
    create_user, verify_user, resolve_user, get_email, set_password,
    create_reset_token, consume_reset_token, count_users,
)
from backend.auth import mailer, google_oauth
from backend.agent import auto_agent
from backend.llm.streaming_provider import get_provider

STATIC = Path(__file__).resolve().parent / "static"


def require_login(request: Request) -> None:
    """Global gate: when ENABLE_AUTH, every non-public route needs a session."""
    if not webauth.auth_enabled() or webauth.is_public_path(request.url.path):
        return
    if not request.session.get("user_id"):
        raise HTTPException(status_code=401, detail="Authentication required")


# ---- Brute-force / abuse protection: simple per-IP sliding-window rate limiter ----
_RATE_BUCKETS: dict = defaultdict(deque)


def _rate_ok(request: Request, name: str, limit: int, window: float = 60.0) -> bool:
    """False when this client IP has exceeded `limit` hits to `name` within `window`s."""
    ip = (request.client.host if request.client else "?") or "?"
    dq = _RATE_BUCKETS[f"{name}:{ip}"]
    now = _time.time()
    while dq and dq[0] < now - window:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


def _too_many():
    return JSONResponse({"error": "Too many attempts. Please wait a minute and try again."},
                        status_code=429)


def _is_loopback(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    return host in ("127.0.0.1", "::1", "localhost")


def _reset_base(request: Request) -> str:
    """Base URL for reset links: an explicit PUBLIC_BASE_URL if set (so an attacker-
    controlled Host header can never poison the link), else the request's base URL."""
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    return base.rstrip("/") if base else str(request.base_url).rstrip("/")


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


@app.get("/reset")
def reset_page():
    return FileResponse(str(STATIC / "reset.html"), headers=_NO_STORE)


# ----------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------
@app.get("/api/me")
def whoami(request: Request):
    if not webauth.auth_enabled():
        return {"auth": False, "user_id": webauth.LOCAL_USER,
                "auto_agent": _auto_agent_enabled()}
    return {
        "auth": True,
        "user_id": request.session.get("user_id"),
        "signup": webauth.signup_enabled(),
        "google": google_oauth.enabled(),
        "auto_agent": _auto_agent_enabled(),
    }


def _adopt_local_sessions(uid: str) -> None:
    """Single-user self-hosted: fold any pre-auth ('local') chats into the signed-in
    account so conversations don't vanish across login-state changes. Gated on a
    single registered user, so multi-user deployments are untouched."""
    try:
        if uid and uid != webauth.LOCAL_USER and count_users() == 1:
            chat_logic.memory().reassign_sessions(webauth.LOCAL_USER, uid)
    except Exception:
        pass


@app.post("/api/login")
def api_login(request: Request, body: dict = Body(default={})):
    if not _rate_ok(request, "login", limit=10):
        return _too_many()
    identifier = (body.get("user_id") or body.get("identifier") or body.get("email") or "").strip()
    pw = body.get("password") or ""
    uid = resolve_user(identifier) or identifier   # accept email OR username
    if verify_user(uid, pw):
        request.session["user_id"] = uid
        _adopt_local_sessions(uid)
        return {"ok": True, "user_id": uid}
    return JSONResponse({"ok": False, "error": "Invalid email/username or password."},
                        status_code=401)


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.post("/api/signup")
def api_signup(request: Request, body: dict = Body(default={})):
    if not _rate_ok(request, "signup", limit=5):
        return _too_many()
    if not webauth.signup_enabled():
        return JSONResponse(
            {"error": "Sign-ups are disabled. Ask an admin to create your account."},
            status_code=403)
    uid = (body.get("user_id") or "").strip()
    pw = body.get("password") or ""
    email = (body.get("email") or "").strip()
    try:
        create_user(uid, pw, email=email)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    request.session["user_id"] = uid
    _adopt_local_sessions(uid)
    return {"ok": True, "user_id": uid}


@app.post("/api/forgot-password")
def api_forgot_password(request: Request, body: dict = Body(default={})):
    """Start a password reset. Always returns a generic message (no account
    enumeration). If the account exists we issue a token and email the link."""
    if not _rate_ok(request, "forgot", limit=5):
        return _too_many()
    generic = {"ok": True,
               "message": "If an account matches, a password-reset link has been sent."}
    identifier = (body.get("identifier") or body.get("user_id") or body.get("email") or "").strip()
    user_id = resolve_user(identifier)
    if not user_id:
        secrets.token_urlsafe(32)   # equalize work vs. the found-account branch (timing)
        return generic
    token = create_reset_token(user_id)
    reset_url = f"{_reset_base(request)}/reset?token={token}"
    email = get_email(user_id)
    sent = mailer.send_email(
        email or "",
        "Reset your Research Assistant password",
        f"Hi {user_id},\n\nUse this link to reset your password (valid for 30 minutes):\n\n"
        f"{reset_url}\n\nIf you didn't request this, you can ignore this email.\n",
    ) if email else False
    if not sent:
        # Only log the token when there is genuinely no email path (self-hosted), so a
        # production SMTP deployment never writes a live bearer token to its logs.
        if not mailer.email_configured():
            print(f"[auth] password-reset link for {user_id!r}: {reset_url}")
        # Self-hosted escape hatch: only ever hand the link back to a SINGLE-USER instance
        # accessed from LOOPBACK — never to a remote or multi-user caller (else knowing a
        # username would be account takeover).
        if (webauth.reset_return_link() and count_users() == 1 and _is_loopback(request)):
            return {**generic, "reset_url": reset_url}
    return generic


@app.post("/api/reset-password")
def api_reset_password(request: Request, body: dict = Body(default={})):
    if not _rate_ok(request, "reset", limit=10):
        return _too_many()
    token = (body.get("token") or "").strip()
    pw = body.get("password") or ""
    # Validate BEFORE consuming the token, so a too-short password doesn't burn the link.
    if len(pw) < 6:
        return JSONResponse({"error": "Password must be at least 6 characters."}, status_code=400)
    user_id = consume_reset_token(token)
    if not user_id:
        return JSONResponse(
            {"error": "This reset link is invalid or has expired. Please request a new one."},
            status_code=400)
    try:
        set_password(user_id, pw)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    request.session.clear()   # force a fresh sign-in with the new password
    return {"ok": True, "user_id": user_id}


def _google_redirect_uri(request: Request) -> str:
    return _reset_base(request) + "/auth/google/callback"


@app.get("/auth/google/login")
def google_login(request: Request):
    if not google_oauth.enabled():
        return RedirectResponse("/login?error=google_off")
    if not _rate_ok(request, "google", limit=10):
        return RedirectResponse("/login?error=rate")
    state = secrets.token_urlsafe(24)
    request.session["oauth_state"] = state
    return RedirectResponse(google_oauth.authorize_url(_google_redirect_uri(request), state))


@app.get("/auth/google/callback")
def google_callback(request: Request, code: str = "", state: str = ""):
    if not google_oauth.enabled():
        return RedirectResponse("/login")
    saved = request.session.pop("oauth_state", None)
    if not code or not state or state != saved:
        return RedirectResponse("/login?error=google")
    try:
        info = google_oauth.exchange_code(code, _google_redirect_uri(request))
    except Exception:
        return RedirectResponse("/login?error=google")
    email = (info.get("email") or "").strip().lower()
    if not email or info.get("email_verified") is False:
        return RedirectResponse("/login?error=google_email")
    # Find the account by email, or create one (email is a valid user_id; the random
    # password is never used — Google users sign in via Google or reset it).
    uid = resolve_user(email)
    if not uid:
        try:
            create_user(email, secrets.token_urlsafe(24), email=email)
            uid = email
        except ValueError:
            return RedirectResponse("/login?error=google")
    request.session["user_id"] = uid
    _adopt_local_sessions(uid)
    return RedirectResponse("/")


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
# Autonomous coding agent (Claude Agent SDK) — write -> run -> test -> fix.
#
# ⚠️  This runs the SDK with Bash/Write/Edit on the HOST (not the Docker sandbox).
# It is an OWNER dev tool, so it is gated hard: OFF by default (ENABLE_AUTO_AGENT),
# callable only from loopback, login-required (middleware), and rate-limited.
# ----------------------------------------------------------------------
def _auto_agent_enabled() -> bool:
    return (os.getenv("ENABLE_AUTO_AGENT", "") or "").strip().lower() in ("1", "true", "yes", "on")


@app.post("/agent/task")
async def agent_task(request: Request, body: dict = Body(default={})):
    if not _auto_agent_enabled():
        return JSONResponse(
            {"error": "The autonomous agent is disabled. Set ENABLE_AUTO_AGENT=true to enable it "
                      "(it runs shell/file tools on the host — owner use only)."},
            status_code=403)
    if not _is_loopback(request):
        return JSONResponse(
            {"error": "The autonomous agent is only callable from localhost (127.0.0.1)."},
            status_code=403)
    if not _rate_ok(request, "agent", limit=5):
        return _too_many()
    task = (body.get("task") or "").strip()
    if not task:
        return JSONResponse({"error": "task is required"}, status_code=400)
    try:
        return await auto_agent.run_auto_agent(task)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:                       # missing key / SDK / CLI
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:                           # pragma: no cover - defensive
        return JSONResponse({"error": f"agent run failed: {exc}"}, status_code=500)


@app.post("/agent/task/stream")
async def agent_task_stream(request: Request, body: dict = Body(default={})):
    """Same as /agent/task but streams each step live as NDJSON (for the in-app UI)."""
    if not _auto_agent_enabled():
        return JSONResponse({"error": "The autonomous agent is disabled. Set ENABLE_AUTO_AGENT=true."},
                            status_code=403)
    if not _is_loopback(request):
        return JSONResponse({"error": "The autonomous agent is only callable from localhost (127.0.0.1)."},
                            status_code=403)
    if not _rate_ok(request, "agent", limit=5):
        return _too_many()
    task = (body.get("task") or "").strip()
    if not task:
        return JSONResponse({"error": "task is required"}, status_code=400)

    async def gen():
        try:
            async for ev in auto_agent.stream_auto_agent(task, str(ROOT)):
                yield json.dumps(ev) + "\n"
        except Exception as exc:                       # pragma: no cover - defensive
            yield json.dumps({"type": "error", "message": f"agent run failed: {exc}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


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
