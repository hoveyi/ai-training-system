@echo off
title AI System - First Time Setup

echo.
echo ================================================
echo      AI System - First Time Setup
echo ================================================
echo.

set PYTHON=C:\Users\DOVE\AppData\Local\Programs\Python\Python313\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo Please edit this file and fix PYTHON path
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
"%PYTHON%" -m pip install streamlit torch torchvision pymysql pandas numpy pillow scikit-learn matplotlib tqdm seaborn sqlalchemy -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo [WARN] Tsinghua mirror failed, trying default...
    "%PYTHON%" -m pip install streamlit torch torchvision pymysql pandas numpy pillow scikit-learn matplotlib tqdm seaborn sqlalchemy
)

echo.
echo [2/3] Initializing MySQL database...
echo Make sure MySQL is running (port 3306)
"%PYTHON%" -c "import sys; sys.path.insert(0, 'streamlist'); from db_config import init_database; init_database(); print('[OK] Database initialized')"
if %errorlevel% neq 0 (
    echo [WARN] DB init failed, check:
    echo   1. Is MySQL running?
    echo   2. Correct username/password in db_config.py?
)

echo.
echo [3/3] Setup complete!
echo.
echo Next: train the 4 models:
echo   cd backend\flower_model
echo   python train.py
echo   cd ..\titanic_model
echo   python train.py
echo   cd ..\fashion_model
echo   python train.py
echo   cd ..\regression_model
echo   python train.py
echo.
echo After training, double-click start.bat to launch
echo ================================================
pause
