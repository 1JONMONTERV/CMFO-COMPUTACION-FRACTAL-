@echo off
TITLE CMFO Compiler & Builder
COLOR 0A

echo =========================================
echo       CMFO NATIVE KERNEL BUILDER
echo =========================================
echo.

:: 1. Check for GCC
where gcc >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] GCC detected. Compiling Native Core...
    
    if not exist "..\..\core\native\bin" mkdir "..\..\core\native\bin"
    
    :: Compile Core
    gcc -O3 -shared -o ..\..\core\native\bin\cmfo_core.dll ..\..\core\native\src\*.c
    
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] cmfo_core.dll compiled successfully!
    ) else (
        COLOR 0C
        echo [ERROR] Compilation failed.
    )
    goto :PYTHON_BUILD
)

:: 2. Check for MSVC (cl)
where cl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] MSVC detected. Compiling Native Core...
     if not exist "..\..\core\native\bin" mkdir "..\..\core\native\bin"
    cl /O2 /LD ..\..\core\native\src\*.c /Fe:..\..\core\native\bin\cmfo_core.dll
    goto :PYTHON_BUILD
)

:: 3. No Compiler Found
COLOR 0E
echo [WARNING] No C Compiler found (GCC or MSVC).
echo To get maximum performance, install MinGW-w64 or Visual Studio C++.
echo.
echo Skipping native compilation...

:PYTHON_BUILD
echo.
echo =========================================
echo       BUILDING PYTHON BINDINGS
echo =========================================
cd ..\bindings\python
python -m pip install -e .

if %ERRORLEVEL% EQU 0 (
    COLOR 0A
    echo.
    echo [SUCCESS] Python package 'cmfo' installed in Editable mode.
    echo You can now use 'import cmfo' in your scripts.
) else (
    COLOR 0C
    echo [ERROR] Python build failed.
)

pause
