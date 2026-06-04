@echo off
REM Remove low-quality / gibberish stored conversations.
REM Usage:
REM   clean_bad_chats.bat               preview only (dry-run)
REM   clean_bad_chats.bat --apply       actually delete bad chats
REM   clean_bad_chats.bat --all         preview deleting ALL chats
REM   clean_bad_chats.bat --all --apply delete ALL chats (still asks YES)
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found.
    exit /b 1
)
if not exist scripts\clean_bad_conversations.py (
    echo ERROR: scripts\clean_bad_conversations.py not found.
    exit /b 1
)

.venv\Scripts\python.exe scripts\clean_bad_conversations.py %*
exit /b %errorlevel%
