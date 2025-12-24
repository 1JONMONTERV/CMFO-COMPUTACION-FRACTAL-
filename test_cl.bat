
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
echo [TEST] CL Location:
where cl
echo [TEST] Compiling minimal C++:
echo int main() { return 0; } > test.cpp
cl test.cpp
