@echo off
echo ==========================================
echo CMFO PRODUCTION DEPLOYMENT
echo ==========================================
echo.

if not exist "bindings\python\dist" (
    echo ERROR: Distribution artifacts not found!
    echo Run build_pypi.bat first.
    exit /b 1
)

echo Found artifacts:
dir bindings\python\dist /b
echo.

echo [1] Verifying Twine is installed...
pip show twine >nul 2>&1
if %errorlevel% neq 0 (
    echo Twine not found. Installing...
    pip install twine
)

echo.
echo [2] Ready to upload to PyPI.
echo.
echo WARNING: This will publish version 0.1.4 to the world.
echo Press Ctrl+C to abort, or any key to proceed.
pause

echo.
echo [3] Uploading...
twine upload bindings\python\dist\*

echo.
if %errorlevel% equ 0 (
    echo SUCCESS: CMFO Deployed to PyPI.
) else (
    echo ERROR: Upload failed. Check credentials.
)
pause
