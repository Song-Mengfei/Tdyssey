@echo off
setlocal
cd /d "%~dp0"
set "TDYSSEY_DEMO_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%TDYSSEY_DEMO_PYTHON%" set "TDYSSEY_DEMO_PYTHON=%~dp0..\Tdyssey\.venv\Scripts\python.exe"
if not exist "%TDYSSEY_DEMO_PYTHON%" (
  echo Existing Python environment was not found. See docs\MANUAL_WORLD_MODEL.md.
  pause
  exit /b 1
)
set SDL_VIDEODRIVER=
set SDL_AUDIODRIVER=dummy
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
"%TDYSSEY_DEMO_PYTHON%" -B scripts\manual_world_model.py %*
if errorlevel 1 pause
