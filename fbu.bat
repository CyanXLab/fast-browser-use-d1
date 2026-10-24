@echo off
rem Convenience shim: forward all arguments to the project venv's fbu CLI.
rem Usage (from a terminal in the project folder):  fbu run <url> --goal "..."
if not exist "%~dp0.venv\Scripts\fbu.exe" (
  echo [fbu] Virtual environment not ready. Run deploy.bat first.
  exit /b 1
)
"%~dp0.venv\Scripts\fbu.exe" %*
