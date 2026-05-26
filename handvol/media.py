try:
    import keyboard as _keyboard
except Exception:
    _keyboard = None

try:
    import pyautogui as _pyautogui
except Exception:
    _pyautogui = None


def play_pause():
    """Send the media play/pause key. Falls back to pyautogui if `keyboard` lacks permission."""
    if _keyboard is not None:
        try:
            _keyboard.send('play/pause media')
            return
        except Exception:
            pass
    if _pyautogui is not None:
        _pyautogui.press('playpause')
        return
    raise RuntimeError("No media key backend available (install `keyboard` or `pyautogui`).")
