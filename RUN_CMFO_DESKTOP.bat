@echo off
title CMFO Desktop Tutor
color 0B
cd /d "%~dp0"

echo.
echo  ================================================================
echo   CMFO DESKTOP TUTOR (GUI Nativa)
echo  ================================================================
echo.
echo   [*] Iniciando aplicacion de escritorio...
echo   [*] Cargando interfaz grafica...
echo.

set "PYTHONPATH=%~dp0"
python d30_desktop_ui\cmfo_desktop.py

if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo [ERROR] La aplicacion fallo.
    echo.
)

pause
