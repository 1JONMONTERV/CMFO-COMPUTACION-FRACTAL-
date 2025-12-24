@echo off
setlocal
echo [SETUP] Buscando compilador C++ (Visual Studio)...

:: Buscar vcvars64.bat en ubicaciones estandar
set "VCVARS="
for /f "delims=" %%f in ('dir /s /b "C:\Program Files\Microsoft Visual Studio\vcvars64.bat" 2^>nul') do (
    set "VCVARS=%%f"
    goto :FOUND
)
for /f "delims=" %%f in ('dir /s /b "C:\Program Files (x86)\Microsoft Visual Studio\vcvars64.bat" 2^>nul') do (
    set "VCVARS=%%f"
    goto :FOUND
)

:FOUND
if "%VCVARS%"=="" (
    echo [ERROR] No se encontro vcvars64.bat. Asegurate de tener VS C++ Build Tools instalado.
    exit /b 1
)

echo [SETUP] Encontrado: "%VCVARS%"
echo [SETUP] Configurando entorno...
call "%VCVARS%"

echo.
echo [BUILD] Compilando Kernel CUDA CMFO...
nvcc -shared -o cmfo_cuda.dll kernel_cmfo.cu
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la compilacion NVCC.
    exit /b 1
)
echo [BUILD] Exito! cmfo_cuda.dll generada.

echo.
echo [RUN] Ejecutando Benchmark Nativo...
python benchmark_cuda_native.py

echo.
echo [DONE] Ciclo tecnico completado.
endlocal
