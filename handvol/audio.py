from ctypes import cast, POINTER

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pycaw.callbacks import MMNotificationClient


_volume_ctrl = None

# Keep the enumerator and notification client alive at module level so they are
# never garbage-collected (GC would silently stop the callbacks from firing).
_enumerator = None
_notification_client = None


class _DefaultDeviceWatcher(MMNotificationClient):
    """Calls _reset_ctrl() when the default render endpoint changes."""

    def on_default_device_changed(self, flow, flow_id, role, role_id, default_device_id):
        # Only the render direction matters; ignore microphone / comms changes.
        if flow == "eRender":
            _reset_ctrl()


def _ensure_watcher():
    """Register the device-change watcher once, on the first _ctrl() call.

    If COM is not available on this thread the registration is skipped
    gracefully — volume control still works via the existing except-based
    _reset_ctrl() fallback.
    """
    global _enumerator, _notification_client
    if _notification_client is not None:
        # Already registered; nothing to do.
        return
    try:
        client = _DefaultDeviceWatcher()
        enumerator = AudioUtilities.GetDeviceEnumerator()
        enumerator.RegisterEndpointNotificationCallback(client)
        # Store both so neither is collected while the app runs.
        _enumerator = enumerator
        _notification_client = client
    except Exception:
        # Graceful degradation: watcher could not be registered; fall through.
        pass


def _ctrl():
    global _volume_ctrl
    if _volume_ctrl is None:
        devices = AudioUtilities.GetSpeakers()
        # Newer pycaw wraps IMMDevice in AudioDevice; the COM Activate lives on ._dev.
        raw = getattr(devices, "_dev", devices)
        interface = raw.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
        # Register the device-change watcher lazily, after we have a working
        # COM apartment (Activate above guarantees COM is available here).
        _ensure_watcher()
    return _volume_ctrl


def _reset_ctrl():
    global _volume_ctrl
    _volume_ctrl = None


def set_volume(percent):
    global _volume_ctrl
    percent = max(0.0, min(100.0, float(percent)))
    try:
        _ctrl().SetMasterVolumeLevelScalar(percent / 100.0, None)
    except Exception:
        _reset_ctrl()
        raise


def get_volume():
    global _volume_ctrl
    try:
        return _ctrl().GetMasterVolumeLevelScalar() * 100.0
    except Exception:
        _reset_ctrl()
        raise


def toggle_mute():
    global _volume_ctrl
    try:
        c = _ctrl()
        c.SetMute(not c.GetMute(), None)
    except Exception:
        _reset_ctrl()
        raise


def is_muted():
    global _volume_ctrl
    try:
        return bool(_ctrl().GetMute())
    except Exception:
        _reset_ctrl()
        raise
