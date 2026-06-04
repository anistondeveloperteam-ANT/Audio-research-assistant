@echo off
REM One-click memory backup. Optional args are passed through.
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found.
    exit /b 1
)
if not exist scripts\export_memory_cli.py (
    echo ERROR: scripts\export_memory_cli.py not found.
    exit /b 1
)

.venv\Scripts\python.exe scripts\export_memory_cli.py %*
exit /b %errorlevel%
