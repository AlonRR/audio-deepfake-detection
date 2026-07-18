@echo off
REM One-click voice-sample recorder.
REM Serves tools/record.html over localhost (getUserMedia needs a secure origin,
REM which file:// does not reliably provide) and opens it in the default browser.

setlocal
set PORT=8731
set HERE=%~dp0

echo Starting recorder on http://localhost:%PORT%/record.html
echo Close this window when you are done recording.
echo.

start "" "http://localhost:%PORT%/record.html"
python -m http.server %PORT% --bind 127.0.0.1 --directory "%HERE%"
