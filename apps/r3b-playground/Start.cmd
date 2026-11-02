@echo off
setlocal
if defined TDYSSEY_PYTHON goto configured
if exist "%~dp0..\..\.venv\Scripts\python.exe" (
  set "TDYSSEY_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\..\Tdyssey\.venv\Scripts\python.exe" (
  set "TDYSSEY_PYTHON=%~dp0..\..\..\Tdyssey\.venv\Scripts\python.exe"
) else (
  set "TDYSSEY_PYTHON=python"
)
:configured
"%TDYSSEY_PYTHON%" -X utf8 -B "%~dp0start.py" --open %*
if errorlevel 1 pause
