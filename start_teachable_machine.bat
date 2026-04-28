@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"
set "PORT=8000"
where py >nul 2>nul
if %errorlevel%==0 set "PY_CMD=py -3"

netstat -ano | find ":8000" >nul
if %errorlevel%==0 set "PORT=8001"

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:%PORT%/'"
%PY_CMD% "Teachable Machine.py" "%PORT%"

if errorlevel 1 (
    echo.
    echo Failed to start Teachable Machine. Check Python and dependencies.
)

pause
