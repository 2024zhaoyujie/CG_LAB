@echo off
cd /d "%~dp0"
if exist "..\work2\.venv\Scripts\python.exe" (
  "..\work2\.venv\Scripts\python.exe" "%~dp0main.py"
) else (
  python "%~dp0main.py"
)
pause
