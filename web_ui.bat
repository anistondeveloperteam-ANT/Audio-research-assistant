@echo off
REM Launch the new web UI (FastAPI) at http://localhost:8600
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found. Create it with: python -m venv .venv ^&^& pip install -r requirements.txt
    exit /b 1
)

echo Open http://localhost:8600 in your browser once it says "Uvicorn running".
.venv\Scripts\python.exe run.py
