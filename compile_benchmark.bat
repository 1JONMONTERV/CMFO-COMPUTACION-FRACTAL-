@echo off
echo [SETUP] Initializing MSVC Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" > nul
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Error calling vcvars64.bat
    exit /b 1
)

echo [INFO] Environment Initialized.
echo [INFO] Running NVCC...
nvcc -shared -o build/sha256_benchmark.dll src/sha256_benchmark.cu -arch=sm_86 -Xcompiler "/wd4819" --allow-unsupported-compiler 2> build_error.txt
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] NVCC Failed.
    type build_error.txt
    exit /b 1
)

echo [SUCCESS] DLL Compiled.
