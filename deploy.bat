@echo off
echo --- CMFO Deployment Tool ---
echo.
echo 1. Adding files to staging...
git add .
if %ERRORLEVEL% NEQ 0 (
    echo Error: Could not add files. Ensure git is installed.
    pause
    exit /b
)

echo.
echo 2. Committing changes...
git commit -m "release: CMFO v1.1.0 Premium Web & Server"
if %ERRORLEVEL% NEQ 0 (
    echo Note: Nothing to commit or error in commit. Continuing...
)

echo.
echo 3. Pushing to GitHub...
git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo --- ERROR: Push Failed ---
    echo Likely causes:
    echo  - You are not logged in.
    echo  - You need to set a Personal Access Token.
    echo.
    echo Try running 'git push' manually in your terminal to see auth prompts.
    pause
    exit /b
)

echo.
echo ------------------------------------------
echo SUCCESS! Deployment complete.
echo Check https://1jonmonterv.github.io/cmfo-universe/ in a few minutes.
echo ------------------------------------------
pause
