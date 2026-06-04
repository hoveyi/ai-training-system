@echo off
title AI System - Starting...

echo.
echo ================================================
echo       AI Integrated Application System
echo ================================================
echo.

set PYTHON=C:\Users\DOVE\AppData\Local\Programs\Python\Python313\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo Please edit this file and fix the PYTHON path
    pause
    exit /b 1
)

echo [1/3] Checking Python...
"%PYTHON%" --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not working
    pause
    exit /b 1
)

echo [2/3] Initializing database...
"%PYTHON%" -c "import sys; sys.path.insert(0, 'streamlist'); from db_config import init_database; init_database(); print('[OK] Database ready')"
if %errorlevel% neq 0 (
    echo [WARN] DB init failed, MySQL may not be running
)

echo [3/3] Starting Streamlit app...
echo.
echo    Opening browser to http://localhost:8501
echo    Press Ctrl+C to stop
echo ================================================
echo.

cd /d "%~dp0streamlist"
"%PYTHON%" -m streamlit run app.py

echo.
echo App stopped, press any key to exit...
pause >nul
