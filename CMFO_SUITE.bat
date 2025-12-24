@echo off
title CMFO FRACTAL SUITE (GPU/CPU HYBRID)
color 0b

echo.
echo ====================================================
echo        CMFO FRACTAL SUITE - OFFICIAL RELEASE
echo ====================================================
echo.
echo [INFO] Inicializando entorno Fractal...
echo [INFO] Detectando Hardware...
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado. Instale Python 3.8+
    pause
    exit
)

:: Launch CLI
echo [OK] Iniciando CLI Principal...
python -m cmfo.apps.cli

pause
