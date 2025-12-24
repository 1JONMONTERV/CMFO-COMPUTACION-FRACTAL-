
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
nvcc -c src/cmfo_kernel.cu -o build/cmfo.obj -arch=sm_86 -ccbin "cl.exe" --allow-unsupported-compiler
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Compliation failed.
    exit /b 1
)
nvcc -shared build/cmfo.obj -o build/cmfo_core.dll
echo [SUCCESS]
