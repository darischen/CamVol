"""Find-or-launch Spotify on Windows.

If Spotify is already running, bring its main window to the foreground without
moving it (no SetWindowPos call, so it stays on whatever monitor the user
parked it on). Otherwise launch via the ``spotify:`` URI scheme so we don't
have to hardcode an install path.
"""
import ctypes
import os
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SW_RESTORE = 9

# Pin the signatures we use — without these ctypes treats HWND as int and the
# 64-bit pointer half can silently get truncated.
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD,
    wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]


def _process_exe(pid):
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        if not kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return None
        return buf.value
    finally:
        kernel32.CloseHandle(h)


def _find_spotify_hwnd():
    """Top-level visible window owned by a Spotify.exe process. Spotify spawns
    several invisible child windows; the one with a non-empty title is the
    user-facing main window."""
    found = []

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindowTextLengthW(hwnd) == 0:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = _process_exe(pid.value)
        if exe and os.path.basename(exe).lower() == "spotify.exe":
            found.append(hwnd)
            return False  # stop enumerating
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return found[0] if found else None


def _force_foreground(hwnd):
    """SetForegroundWindow is blocked unless the caller already owns input
    focus (anti-flicker policy). Workaround: attach our thread's input queue
    to the current foreground's, perform the focus change, detach."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    fg = user32.GetForegroundWindow()
    fg_tid = user32.GetWindowThreadProcessId(fg, None)
    my_tid = kernel32.GetCurrentThreadId()

    attached = False
    if fg_tid and fg_tid != my_tid:
        attached = bool(user32.AttachThreadInput(my_tid, fg_tid, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(my_tid, fg_tid, False)


def focus_or_launch():
    """Bring Spotify to focus if running; launch it otherwise. Returns one of
    'focused', 'launched', 'failed' for logging."""
    hwnd = _find_spotify_hwnd()
    if hwnd:
        try:
            _force_foreground(hwnd)
            return "focused"
        except Exception:
            return "failed"
    try:
        # ShellExecute via os.startfile invokes the registered handler for the
        # spotify: URI scheme — same path Spotify uses for click-to-play links.
        os.startfile("spotify:")
        return "launched"
    except OSError:
        return "failed"
