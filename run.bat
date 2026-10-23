@echo off
setlocal
cd /d "%~dp0"

rem Keep Python's temporary wheel files inside the project workspace.
if not exist .tmp mkdir .tmp
set "TEMP=%CD%\.tmp"
set "TMP=%CD%\.tmp"

if not exist .venv\Scripts\python.exe python -m venv .venv
if errorlevel 1 (echo Could not create the Python environment. Install Python 3.10 or newer and try again.& pause& exit /b 1)

.venv\Scripts\python.exe -m pip --version >nul 2>&1
if errorlevel 1 .venv\Scripts\python.exe -m ensurepip --upgrade
if errorlevel 1 (echo Could not initialize pip in the Python environment.& pause& exit /b 1)

.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. Check your internet connection and try again.& pause& exit /b 1)

.venv\Scripts\python.exe -u server.py
pause
