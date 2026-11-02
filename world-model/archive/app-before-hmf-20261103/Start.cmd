@echo off
"%~dp0..\..\..\Tdyssey\.venv\Scripts\python.exe" -X utf8 -B "%~dp0start.py" --open
if errorlevel 1 pause
