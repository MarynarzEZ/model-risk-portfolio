@echo off
REM Portfolio Risk Desk -- double-click launcher.
REM Opens the app in the default browser. Close this window to stop it.

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

echo.
echo   Portfolio Risk Desk
echo   -------------------
echo   Otwieram http://localhost:8501 w przegladarce.
echo   Zamknij to okno, zeby zatrzymac aplikacje.
echo.

REM The browser is opened from a parallel process a few seconds from now,
REM because the server below blocks this one until it is stopped.
REM
REM The server runs headless on purpose. Without it, Streamlit asks for
REM an e-mail address on its very first run and waits for an answer at a
REM terminal nobody is looking at, so a double-click would appear to hang.
REM Headless skips that prompt, at the cost of opening the browser here.
start "" /B powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false

echo.
echo  Aplikacja zostala zatrzymana.
echo  Jesli powodem byl blad, jego tresc jest powyzej.
echo  Czesty powod: port 8501 jest juz zajety przez inna kopie aplikacji.
pause
