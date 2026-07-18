@echo off
REM Local web UI for the voice cloner + deepfake detector.
REM Needs: the lab VPN up, and `ssh shenkar` working (see docs/server_runbook.md).

setlocal
cd /d "%~dp0..\.."

echo Starting the voice-clone / detector UI on http://localhost:7860
echo (VPN must be up and `ssh shenkar` must work.)
echo Close this window to stop.
echo.

start "" "http://localhost:7860"
uv run --with flask python tools/webui/app.py
