@echo off
REM Build / refresh the research paper search index.
REM   build_index.bat                full rebuild (all papers)
REM   build_index.bat --incremental  only changed PDFs
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found. Create it with: python -m venv .venv ^&^& pip install -r requirements.txt
    exit /b 1
)

.venv\Scripts\python.exe pipeline.py %*
exit /b %errorlevel%
