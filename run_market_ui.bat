@echo off
REM Launch the Audio Research market UI on port 8501.
setlocal enableextensions
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv\Scripts\python.exe not found. Create it with: python -m venv .venv ^&^& pip install -r requirements.txt
    exit /b 1
)
if not exist frontend\streamlit_app.py (
    echo ERROR: frontend\streamlit_app.py not found.
    exit /b 1
)

REM Default ingestion / retrieval tuning (overridable via .env)
if "%PARSER_MODE%"=="" set PARSER_MODE=auto
if "%ANSWER_PROVIDER%"=="" set ANSWER_PROVIDER=manual

.venv\Scripts\python.exe -m py_compile frontend\streamlit_app.py
if errorlevel 1 (
    echo UI compile failed. See the error above.
    exit /b 1
)

echo Starting Market UI on http://localhost:8501  (Ctrl+C to stop)
.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py --server.port 8501
