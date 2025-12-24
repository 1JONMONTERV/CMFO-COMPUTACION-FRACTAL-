
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

echo [STEP 1] Compile to Object
nvcc -c src/cmfo_kernel.cu -o build/cmfo.obj -arch=sm_86
if %ERRORLEVEL% NEQ 0 goto fail

echo [STEP 2] Link to DLL (using NVCC default linker)
nvcc -shared build/cmfo.obj -o build/cmfo_core.dll
if %ERRORLEVEL% NEQ 0 goto fail

echo [SUCCESS]
exit /b 0

:fail
echo [FAIL]
exit /b 1
