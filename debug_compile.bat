
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
where cl
where nvcc
nvcc src/cmfo_kernel.cu -o build/test.dll --shared
