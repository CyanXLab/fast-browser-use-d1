@echo off
rem Fast Browser Use - Windows one-click deploy launcher.
rem Right-click -> "Run with PowerShell" is NOT needed; double-click works.
rem The heavy lifting lives in deploy.ps1 (same folder).
setlocal
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
where powershell >nul 2>nul
if errorlevel 1 (
  echo [deploy] Windows PowerShell not found. Please install Windows PowerShell 5.1+.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%deploy.ps1" %*
set "EC=%ERRORLEVEL%"
echo.
echo [deploy] Deploy script finished with exit code %EC%.
pause
exit /b %EC%
