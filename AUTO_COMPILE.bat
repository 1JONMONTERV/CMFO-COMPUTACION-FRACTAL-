@echo off
echo [AUTO] Setting up environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

echo [AUTO] Compiling CUDA Kernel...
nvcc -shared -o cmfo_cuda.dll kernel_cmfo.cu
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Compilation failed.
    exit /b 1
)

echo [AUTO] Running Native Benchmark...
python benchmark_cuda_native.py
