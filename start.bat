@echo off
REM Launches the risk desk in the default browser.
REM Double-click this file; no terminal, no commands to remember.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  Nie znaleziono srodowiska .venv w tym folderze.
    echo.
    echo  Utworz je raz, w terminalu:
    echo      python -m venv .venv
    echo      .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Uruchamiam Portfolio Risk Desk...
echo Zamknij to okno, zeby zatrzymac aplikacje.
echo.

REM --server.headless stops Streamlit printing its "enter your email"
REM first-run prompt, which blocks startup when nobody is at a terminal.
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

if errorlevel 1 (
    echo.
    echo  Aplikacja zakonczyla sie bledem. Tresc bledu jest powyzej.
    pause
)
