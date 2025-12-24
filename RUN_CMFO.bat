@echo off
title CMFO SOVEREIGN TUTOR - DEBUG MODE
color 0A
cd /d "%~dp0"

echo.
echo  ================================================================
echo   CMFO LAUNCHER (ROBUST)
echo  ================================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in global PATH.
    echo Please install Python 3.10+ and add to PATH.
    pause
    exit /b
)

:: 2. Set Env
set "PYTHONPATH=%~dp0"
echo [INFO] PYTHONPATH: %PYTHONPATH%
echo [INFO] PORT: 8088

echo.
echo [INFO] Starting Server...
echo [INFO] IF THIS CRASHES, SEE 'launch_error.log' below.
echo.

:: 3. Run
echo [INFO] Launching Browser...
start "" "http://localhost:8088"
python d29_edu_ui\server.py 2> launch_error.log

:: 4. Check Exit
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo [FATAL ERROR] The server crashed.
    echo -----------------------------------------------------------
    type launch_error.log
    echo -----------------------------------------------------------
    echo.
)

pause
