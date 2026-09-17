@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-runtime\Scripts\python.exe" (
    echo Preparing the JARVIS runtime environment...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo Setup failed. Install 64-bit CPython 3.13 with Tcl/Tk, then try again.
        pause
        exit /b 1
    )
)

"%~dp0.venv-runtime\Scripts\python.exe" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo JARVIS stopped with an error. Run main.py --self-test for details.
    pause
)
