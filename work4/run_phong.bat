@echo off
cd /d "%~dp0"
REM 使用 uv：自动选兼容版本（3.10~3.12）并安装 taichi 后运行
where uv >nul 2>&1
if errorlevel 1 (
  echo Please install uv: https://docs.astral.sh/uv/getting-started/
  pause
  exit /b 1
)
uv sync
uv run python phong_raytracing.py
if errorlevel 1 pause
pause
