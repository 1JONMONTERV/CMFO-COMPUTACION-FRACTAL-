@echo off
TITLE CMFO C++ Matrix Engine Builder

echo ==============================================
echo       CMFO C++ MATRIX ENGINE BUILDER
echo ==============================================

:: 1. Try GCC (MinGW)
where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] GCC detected. Compiling...
    g++ -O3 -std=c++17 -o ..\core\language\test_matrix.exe ..\core\language\matrix_engine.cpp ..\core\language\test_matrix.cpp
    goto :CHECK_BUILD
)

:: 2. Try MSVC (cl)
where cl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] MSVC detected in PATH. Compiling...
    cl /O2 /EHsc /Fe:..\core\language\test_matrix.exe ..\core\language\matrix_engine.cpp ..\core\language\test_matrix.cpp
    goto :CHECK_BUILD
)

:: 3. Try to find Visual Studio Environment
echo [INFO] Compiler not in PATH. Searching for Visual Studio...
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VS_PATH%" (
    echo [INFO] Found VS 2022. Setting up environment...
    call "%VS_PATH%"
    cl /O2 /EHsc /Fe:..\core\language\test_matrix.exe ..\core\language\matrix_engine.cpp ..\core\language\test_matrix.cpp
    goto :CHECK_BUILD
)

echo [ERROR] No C++ Compiler found. Please install Visual Studio 2022 or MinGW.
exit /b 1

:CHECK_BUILD
if exist "..\core\language\test_matrix.exe" (
    echo [SUCCESS] Compilation complete. Starting Test...
    echo.
    ..\core\language\test_matrix.exe
) else (
    echo [ERROR] Build failed.
)

pause
