@echo off
REM Restore memory from a backup bundle.
REM Usage: import_memory.bat <bundle.tar.gz> [--mode merge^|replace] [--dry-run]
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found.
    exit /b 1
)
if not exist scripts\import_memory_cli.py (
    echo ERROR: scripts\import_memory_cli.py not found.
    exit /b 1
)
if "%~1"=="" (
    echo Usage: import_memory.bat ^<bundle.tar.gz^> [--mode merge^|replace] [--dry-run]
    exit /b 1
)

.venv\Scripts\python.exe scripts\import_memory_cli.py %*
exit /b %errorlevel%
