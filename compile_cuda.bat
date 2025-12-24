@echo off
echo [BUILD] Compilando CMFO CUDA Kernel Nativo...

REM Configurar entorno de Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se pudo inicializar Visual Studio 2022
    exit /b 1
)

REM Verificar compiladores
echo [CHECK] Verificando compiladores...
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe no encontrado
    exit /b 1
)
echo [OK] cl.exe encontrado

where nvcc >nul 2>&1
if errorlevel 1 (
    set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
)

REM Crear directorio build
if not exist "build" mkdir build

echo [BUILD] Compilando src/cmfo_kernel.cu...
echo [INFO] Arquitectura: sm_86 (RTX 3050)
echo.

nvcc src/cmfo_kernel.cu -o build/cmfo_core.dll --shared -arch=sm_86 -O3 --use_fast_math

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] build/cmfo_core.dll generado exitosamente!
    dir build\cmfo_core.dll
    exit /b 0
) else (
    echo.
    echo [FAIL] Error en compilacion CUDA.
    exit /b 1
)
