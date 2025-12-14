@echo off
echo ========================================
echo CMFO v1.0.0 - Test Suite Runner
echo ========================================

REM Set python path to include bindings/python
set PYTHONPATH=%CD%\bindings\python

echo Running Demo...
python examples\demo_v1.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Demo failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Running Unit Tests...
python -m pytest tests -v
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some unit tests failed.
    echo detailed output above.
) else (
    echo [SUCCESS] All tests passed.
)

pause
