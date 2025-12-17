@echo off
title INSTALADOR CMFO FRACTAL SUITE
color 0a

echo ====================================
echo   INSTALANDO DEPENDENCIAS CMFO
echo ====================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no detectado. Instale Python y agreguelo al PATH.
    pause
    exit
)

echo [1/3] Actualizando PIP...
python -m pip install --upgrade pip

echo [2/3] Instalando Librerias (NumPy, Pillow)...
pip install -r requirements.txt

echo.
echo [3/3] Verificando Instalacion...
python -c "import numpy; import PIL; print('   [OK] Todo listo.')"

echo.
echo ====================================
echo   INSTALACION COMPLETADA
echo ====================================
echo Ahora puede ejecutar CMFO_SUITE.bat
pause
