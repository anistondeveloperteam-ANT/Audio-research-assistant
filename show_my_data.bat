@echo off
REM Inspect indexed PDFs, chunks, embeddings, and memory. READ-ONLY.
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found.
    exit /b 1
)
if not exist viewer_tool\show_my_data.py (
    echo ERROR: viewer_tool\show_my_data.py not found.
    exit /b 1
)

.venv\Scripts\python.exe viewer_tool\show_my_data.py
exit /b %errorlevel%
