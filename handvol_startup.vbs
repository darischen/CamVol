' HandVol silent launcher.
' Drop a shortcut to this file in:
'   shell:startup   (per-user)  -->  %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
' or run it directly to test.
'
' Edit PROJECT_DIR below if you move the repo, or if your venv lives elsewhere.

Option Explicit

Dim PROJECT_DIR, PYTHONW
PROJECT_DIR = "C:\Users\daris\Desktop\School\CamVol"

' Prefer a venv's pythonw.exe if present, else fall back to whatever pythonw is on PATH.
Dim fso : Set fso = CreateObject("Scripting.FileSystemObject")
If fso.FileExists(PROJECT_DIR & "\.venv\Scripts\pythonw.exe") Then
    PYTHONW = """" & PROJECT_DIR & "\.venv\Scripts\pythonw.exe"""
ElseIf fso.FileExists(PROJECT_DIR & "\venv\Scripts\pythonw.exe") Then
    PYTHONW = """" & PROJECT_DIR & "\venv\Scripts\pythonw.exe"""
Else
    PYTHONW = "pythonw.exe"
End If

Dim sh : Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = PROJECT_DIR
' 0 = hidden window, False = do not wait for the process to exit.
sh.Run PYTHONW & " -m handvol.main --headless", 0, False
