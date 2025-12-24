
@echo off
echo [BUILD] Buscando compilar cmfo_core.dll para CMFO Benchmark...

@REM Force MSVC Environment
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

@REM Intentar localizar NVCC si no esta en PATH
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" (
    set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
)
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" (
    set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
)

if not exist "build" mkdir build

echo [BUILD] Estado del entorno:
where cl
where nvcc

echo [BUILD] Compilando src/cmfo_kernel.cu (VERBOSE)...
nvcc -v src/cmfo_kernel.cu -o build/cmfo_core.dll --shared -arch=sm_86

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] build/cmfo_core.dll generado con exito.
    exit /b 0
) else (
    echo [FAIL] Error en compilacion.
    exit /b 1
)
