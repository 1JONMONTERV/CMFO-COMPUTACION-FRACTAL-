
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if exist "build\minimal.dll" del "build\minimal.dll"
nvcc src/minimal.cu -o build/minimal.dll --shared --allow-unsupported-compiler -arch=sm_86
if exist "build\minimal.dll" (
    echo [SUCCESS] Minimal DLL built.
) else (
    echo [FAIL] Build failed.
)
