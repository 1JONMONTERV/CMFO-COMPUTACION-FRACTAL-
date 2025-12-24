
@echo off
setlocal

echo [INIT] Setting up Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

echo [INIT] Checking NVCC...
nvcc --version

if not exist "build" mkdir build

echo [BUILD] Launching NVCC...
nvcc -v src/cmfo_kernel.cu -o build/cmfo_core.dll --shared -arch=sm_86 --allow-unsupported-compiler -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"

if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Compilation failed.
    exit /b 1
)

echo [SUCCESS] DLL compiled at build/cmfo_core.dll
exit /b 0
