"""Send system shortcuts via pyautogui.

Alt+F4 for close window, Ctrl+Shift+Esc for Task Manager.
Follows the same pattern as taskbar.focus_slot() for robustness.
"""
import time
import pyautogui

pyautogui.PAUSE = 0

MODIFIER_WARMUP = 0.05
INTER_KEY_DELAY = 0.05


def close_window():
    """Send Alt+F4 to close the active window. Returns 'ok' or 'failed'."""
    try:
        pyautogui.keyDown("alt")
        try:
            time.sleep(MODIFIER_WARMUP)
            pyautogui.press("f4")
        finally:
            time.sleep(INTER_KEY_DELAY)
            pyautogui.keyUp("alt")
        return "ok"
    except Exception:
        return "failed"


def open_task_manager():
    """Send Ctrl+Shift+Esc to open Task Manager. Returns 'ok' or 'failed'."""
    try:
        pyautogui.keyDown("ctrl")
        try:
            time.sleep(MODIFIER_WARMUP)
            pyautogui.keyDown("shift")
            time.sleep(INTER_KEY_DELAY)
            pyautogui.press("esc")
        finally:
            time.sleep(INTER_KEY_DELAY)
            pyautogui.keyUp("shift")
            time.sleep(INTER_KEY_DELAY)
            pyautogui.keyUp("ctrl")
        return "ok"
    except Exception:
        return "failed"
