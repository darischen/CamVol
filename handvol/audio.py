from ctypes import cast, POINTER

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


_volume_ctrl = None


def _ctrl():
    global _volume_ctrl
    if _volume_ctrl is None:
        devices = AudioUtilities.GetSpeakers()
        # Newer pycaw wraps IMMDevice in AudioDevice; the COM Activate lives on ._dev.
        raw = getattr(devices, "_dev", devices)
        interface = raw.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    return _volume_ctrl


def set_volume(percent):
    percent = max(0.0, min(100.0, float(percent)))
    _ctrl().SetMasterVolumeLevelScalar(percent / 100.0, None)


def get_volume():
    return _ctrl().GetMasterVolumeLevelScalar() * 100.0


def toggle_mute():
    c = _ctrl()
    c.SetMute(not c.GetMute(), None)


def is_muted():
    return bool(_ctrl().GetMute())
