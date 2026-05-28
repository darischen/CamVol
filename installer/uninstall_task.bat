@echo off
REM Task Scheduler cleanup for HandVol.
REM Removes the scheduled task created by install_task.bat. Called by the
REM NSIS uninstaller. Exits 0 even if the task is already gone so uninstall
REM never blocks on a missing task.

setlocal

schtasks /delete /tn "HandVol" /f

if errorlevel 1 (
    echo Task Scheduler entry not found (already removed) - continuing
    exit /b 0
)

echo Task Scheduler entry removed successfully
exit /b 0
