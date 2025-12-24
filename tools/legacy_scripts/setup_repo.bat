@echo off
TITLE CMFO Repository Setup (Auto-Magic)
COLOR 0A

echo =====================================================
echo          CMFO REPOSITORY SETUP ASSISTANT
echo =====================================================
echo.

:: 1. Check if GIT is available
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    COLOR 0E
    echo [WARNING] Git command not found.
    echo.
    echo But don't worry! I detected you have Windows Package Manager (winget).
    echo I can install Git for you automatically right now.
    echo.
    echo Press any key to INSTALL GIT automatically...
    pause >nul
    
    echo.
    echo Installing Git... (This might take a minute)
    winget install -e --id Git.Git
    
    echo.
    echo ---------------------------------------------------
    echo [IMPORTANT] Git Installed!
    echo.
    echo Windows needs to refresh to see the new command.
    echo Please CLOSE this window and DOUBLE CLICK IT AGAIN.
    echo ---------------------------------------------------
    pause
    exit /b
)

echo [OK] Git is ready.
echo.
echo ---------------------------------------
echo 1. Initializing Git...
git init

echo 2. Adding Remote...
git remote remove origin 2>nul
git remote add origin https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-.git

echo 3. Creating Main Branch...
git branch -M main

echo 4. Staging Files...
git add .

echo 5. Committing...
git commit -m "feat: Initial commit of Consolidated CMFO Monorepo v1.1.0"

echo 6. Pushing to GitHub...
echo.
echo [ATTENTION] A window might pop up asking you to sign in to GitHub.
echo Please sign in if asked.
echo.
git push -u origin main

if %ERRORLEVEL% NEQ 0 (
    COLOR 0C
    echo.
    echo [ERROR] Push Failed.
    echo Check your internet or if the repository URL is correct.
) else (
    COLOR 0A
    echo.
    echo [SUCCESS] CMFO Universe has been uploaded!
)

echo.
echo Press any key to finish.
pause >nul
