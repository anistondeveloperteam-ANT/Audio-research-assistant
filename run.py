#!/usr/bin/env python
"""
run.py -- launch the Audio Research Assistant web app.

Usage:
    python run.py                # Chat UI    (Streamlit)  http://localhost:8502  (default)
    python run.py --web          # New web UI (FastAPI)    http://localhost:8600
    python run.py --market       # Market UI  (Streamlit)  http://localhost:8501
    python run.py --dashboard    # Quality dashboard       http://localhost:8503
    python run.py --port 9000    # override the port

The Streamlit apps and the new FastAPI web UI both load config from the local .env.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

APPS = {
    "chat":      (ROOT / "frontend" / "chat_ui.py", 8502),
    "market":    (ROOT / "frontend" / "market_ui.py", 8501),
    "dashboard": (ROOT / "frontend" / "quality_dashboard.py", 8503),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Audio Research Assistant app.")
    parser.add_argument("--web", action="store_true",
                        help="Launch the new FastAPI web UI (http://localhost:8600).")
    parser.add_argument("--market", action="store_true",
                        help="Launch the Market UI instead of the Chat UI.")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the retrieval-quality dashboard.")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the server port.")
    args = parser.parse_args()

    if args.web:
        port = args.port or 8600
        cmd = [sys.executable, "-m", "uvicorn", "webapp.server:app", "--port", str(port)]
        print(f"Starting web UI on http://localhost:{port}  (Ctrl+C to stop)")
        try:
            return subprocess.call(cmd, cwd=str(ROOT))
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0

    app_key = "dashboard" if args.dashboard else "market" if args.market else "chat"
    app_path, default_port = APPS[app_key]
    port = args.port or default_port

    if not app_path.exists():
        print(f"ERROR: {app_path} not found.", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
    ]
    print(f"Starting {app_key} UI on http://localhost:{port}  (Ctrl+C to stop)")
    # Run from the project root so `import backend.*` resolves inside the app.
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        # User pressed Ctrl+C — Streamlit already shut itself down. Exit quietly.
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
