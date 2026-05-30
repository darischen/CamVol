; HandVol Windows Installer
; Built with NSIS 3.x
;
; Path note: makensis resolves relative File/OutFile paths against THIS script's
; directory (installer/), not the working directory. To avoid that trap,
; build_installer.py passes the absolute repo root as a define:
;     makensis /DPROJ_ROOT=<abs repo root> handvol-installer.nsi
; and every source/output path below is built from ${PROJ_ROOT}.

!ifndef PROJ_ROOT
  !error "PROJ_ROOT is not defined. Build via build_installer.py (it passes /DPROJ_ROOT)."
!endif

!ifndef VERSION
  !define VERSION "0.0.0-dev"
!endif

!include "MUI2.nsh"
!include "x64.nsh"
!include "nsDialogs.nsh"
!include "LogicLib.nsh"

; Name and file
Name "HandVol"
OutFile "${PROJ_ROOT}\dist\HandVol-${VERSION}-Installer.exe"
InstallDir "$PROGRAMFILES64\HandVol"
InstallDirRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "InstallLocation"

; Request admin: writing to Program Files and creating a Task Scheduler entry
; both require elevation.
RequestExecutionLevel admin

; MUI Settings
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

; Custom finish page with checkboxes (run now / desktop shortcut)
Page custom FinishPage FinishPageLeave

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; Variables for the custom finish page
Var runNow
Var createShortcut

; ============================================================
; Installer Sections
; ============================================================

Section "Install"
  SetOutPath "$INSTDIR"

  ; Extract bundled files staged by build_installer.py under build/.
  File /r "${PROJ_ROOT}\build\python"
  File /r "${PROJ_ROOT}\build\handvol"
  File /r "${PROJ_ROOT}\build\models"
  File "${PROJ_ROOT}\build\handvol.pyw"
  File "${PROJ_ROOT}\build\requirements.txt"

  ; Task Scheduler helper scripts.
  File "${PROJ_ROOT}\installer\install_task.bat"
  File "${PROJ_ROOT}\installer\uninstall_task.bat"

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Register in Add/Remove Programs
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "DisplayName" "HandVol"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "Publisher" "HandVol"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "NoRepair" 1

  ; Create the Task Scheduler entry. Pass INSTDIR so the batch script can build
  ; absolute paths to pythonw.exe and handvol.pyw.
  ExecWait '"$SYSDIR\cmd.exe" /c ""$INSTDIR\install_task.bat" "$INSTDIR""'

SectionEnd

; ============================================================
; Custom Finish Page
; ============================================================

Function FinishPage
  !insertmacro MUI_HEADER_TEXT "Installation Complete" "HandVol is installed. Choose what to do next."

  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  ${NSD_CreateCheckbox} 0 10u 100% 12u "Run HandVol now"
  Pop $runNow
  ${NSD_SetState} $runNow ${BST_CHECKED}

  ${NSD_CreateCheckbox} 0 28u 100% 12u "Create a desktop shortcut"
  Pop $createShortcut
  ${NSD_SetState} $createShortcut ${BST_CHECKED}

  nsDialogs::Show
FunctionEnd

Function FinishPageLeave
  ; Read checkbox states as the user leaves the page (clicks the wizard's
  ; "Finish" button), then act on them.
  ${NSD_GetState} $runNow $0
  ${NSD_GetState} $createShortcut $1

  ${If} $1 == ${BST_CHECKED}
    CreateShortCut "$DESKTOP\HandVol.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\handvol.pyw"' "$INSTDIR\python\pythonw.exe" 0
  ${EndIf}

  ${If} $0 == ${BST_CHECKED}
    Exec '"$INSTDIR\python\pythonw.exe" "$INSTDIR\handvol.pyw"'
  ${EndIf}
FunctionEnd

; ============================================================
; Uninstaller
; ============================================================

Section "Uninstall"
  ; Remove Task Scheduler entry
  ExecWait '"$SYSDIR\cmd.exe" /c ""$INSTDIR\uninstall_task.bat""'

  ; Best-effort: stop a running instance so files aren't locked.
  ExecWait '"$SYSDIR\taskkill.exe" /f /im pythonw.exe'

  ; Remove desktop shortcut if it exists
  Delete "$DESKTOP\HandVol.lnk"

  ; Remove registry entries
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol"

  ; Remove all installed files
  RMDir /r "$INSTDIR"
SectionEnd
