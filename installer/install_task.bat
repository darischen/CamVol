@echo off
REM Task Scheduler setup for HandVol.
REM Called by the NSIS installer as:  install_task.bat "<INSTDIR>"

setlocal

REM %~1 strips the surrounding quotes the installer passes around INSTDIR.
set "INSTDIR=%~1"

if "%INSTDIR%"=="" (
    echo Error: INSTDIR not provided
    exit /b 1
)

set "PYTHONW=%INSTDIR%\python\pythonw.exe"
set "SCRIPT=%INSTDIR%\handvol.pyw"

REM schtasks /tr needs the program path quoted (Program Files has a space), and
REM the whole /tr value quoted too. The \" sequences are how you embed quotes
REM inside that outer-quoted value from a batch file.
REM
REM   /tn   task name
REM   /tr   command to run (pythonw.exe + the script)
REM   /sc   ONLOGON = when the user logs in. HandVol is a system-tray GUI that
REM         needs the interactive desktop, webcam, and the user's audio session,
REM         so it must run in the user session (not the Session 0 that ONSTART
REM         would use).
REM   /rl   HIGHEST = run elevated (needed for audio control)
REM   /f    overwrite an existing task without prompting
schtasks /create /tn "HandVol" /tr "\"%PYTHONW%\" \"%SCRIPT%\"" /sc onlogon /rl highest /f

if errorlevel 1 (
    echo Error creating Task Scheduler entry
    exit /b 1
)

echo Task Scheduler entry created successfully
exit /b 0
