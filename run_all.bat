@echo off
REM ======================================================
REM  EDB orchestrator - full run
REM  Double-click this file to run everything
REM ======================================================

cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

REM Activate venv if present
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found on PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

python run_all.py %*

echo.
pause
