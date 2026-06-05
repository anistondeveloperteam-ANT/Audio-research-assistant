#!/usr/bin/env python
"""
run.py -- launch the Audio Research Assistant web app (FastAPI).

Usage:
    python run.py                # http://localhost:8600
    python run.py --port 9000    # override the port

A thin wrapper around uvicorn. Config is read from the local .env.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Audio Research Assistant web app.")
    parser.add_argument("--port", type=int, default=8600, help="Server port (default: 8600).")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "uvicorn", "webapp.server:app", "--port", str(args.port)]
    print(f"Starting web UI on http://localhost:{args.port}  (open it in your browser; Ctrl+C to stop)")
    # Run from the project root so `import backend.*` / `import webapp.*` resolve.
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
