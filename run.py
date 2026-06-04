#!/usr/bin/env python
"""
run.py -- launch the Audio Research Assistant web app.

Usage:
    python run.py                # Chat UI   on http://localhost:8502  (default)
    python run.py --market       # Market UI on http://localhost:8501
    python run.py --port 8600     # override the port

This is a thin wrapper around `streamlit run`. The Streamlit app loads
configuration from the local .env file.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

APPS = {
    "chat":   (ROOT / "frontend" / "chat_ui.py", 8502),
    "market": (ROOT / "frontend" / "streamlit_app.py", 8501),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Audio Research Assistant app.")
    parser.add_argument("--market", action="store_true",
                        help="Launch the Market UI instead of the Chat UI.")
    parser.add_argument("--port", type=int, default=None,
                        help="Override the server port.")
    args = parser.parse_args()

    app_key = "market" if args.market else "chat"
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
