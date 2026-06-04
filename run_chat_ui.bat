@echo off
REM Launch the Audio Research chat UI on port 8502.
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found. Create it with: python -m venv .venv ^&^& pip install -r requirements.txt
    exit /b 1
)
if not exist frontend\chat_ui.py (
    echo ERROR: frontend\chat_ui.py not found.
    exit /b 1
)

.venv\Scripts\python.exe run.py
