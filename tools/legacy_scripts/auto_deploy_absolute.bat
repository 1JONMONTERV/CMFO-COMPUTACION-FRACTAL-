@echo off
TITLE FORCE DEPLOY CMFO
COLOR 0A

:: 1. Go to the EXACT folder regardless of where this file is
cd /d "C:\Users\UNSC\.gemini\antigravity\scratch\CMFO-COMPUTACION-FRACTAL"

echo Working in: %CD%
echo.

:: 2. Initialize and Stage
if not exist ".git" (
    echo Initializing Git...
    git init
)

echo Adding files...
git add .

echo Committing...
git commit -m "feat: Consolidate CMFO Universe v1.1.0"

:: 3. Setup Remote (Force reset if exists)
echo Setting remote...
git remote remove origin 2>nul
git remote add origin https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-.git

:: 4. Push
echo.
echo PUSHING TO GITHUB...
echo.
git branch -M main
git push -u origin main

echo.
echo IF YOU SEE "Success" or data transfer above, IT WORKED!
pause
