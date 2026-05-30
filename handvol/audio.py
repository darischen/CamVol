from ctypes import cast, POINTER

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pycaw.callbacks import MMNotificationClient


_volume_ctrl = None

# Keep the enumerator and notification client alive at module level so they are
# never garbage-collected (GC would silently stop the callbacks from firing).
_enumerator = None
_notification_client = None

# Sentinel: True once _ensure_watcher() has been attempted (success or failure).
# Prevents expensive cross-process COM calls (GetDeviceEnumerator +
# RegisterEndpointNotificationCallback) from being retried on every _ctrl()
# invocation when registration permanently fails.
_watcher_attempted = False


class _DefaultDeviceWatcher(MMNotificationClient):
    """Calls _reset_ctrl() when the default render endpoint changes.

    This callback fires on pycaw's COM/MTA thread.  The bare assignment in
    _reset_ctrl() is intentionally lock-free: under the GIL, a single pointer
    store is atomic, so there is no corruption risk.  If the gesture worker
    thread happens to receive a just-invalidated pointer for one frame the next
    COM call raises COMError, the existing except path calls _reset_ctrl() and
    the caller retries on the following frame — i.e. at most one dropped frame,
    self-healing with no added locking overhead.
    """

    def on_default_device_changed(self, flow, flow_id, role, role_id, default_device_id):
        # Only the render direction matters; ignore microphone / comms changes.
        if flow == "eRender":
            _reset_ctrl()


def _ensure_watcher():
    """Register the device-change watcher once, on the first _ctrl() call.

    If COM is not available on this thread the registration is skipped
    gracefully — volume control still works via the existing except-based
    _reset_ctrl() fallback.  The attempt is made at most once for the app's
    lifetime: once _watcher_attempted is True the function returns immediately,
    avoiding repeated cross-process COM calls when registration fails permanently.
    """
    global _enumerator, _notification_client, _watcher_attempted
    if _watcher_attempted:
        # Already attempted (success or failure); nothing to do.
        return
    _watcher_attempted = True
    try:
        client = _DefaultDeviceWatcher()
        enumerator = AudioUtilities.GetDeviceEnumerator()
        enumerator.RegisterEndpointNotificationCallback(client)
        # Store both so neither is collected while the app runs.
        _enumerator = enumerator
        _notification_client = client
    except Exception:
        # Graceful degradation: watcher could not be registered; fall through.
        # Volume control continues to work via the except-based fallback in
        # set_volume / get_volume / toggle_mute / is_muted.
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
    # Intentionally lock-free — see _DefaultDeviceWatcher docstring for rationale.
    global _volume_ctrl
    _volume_ctrl = None


def set_volume(percent):
    percent = max(0.0, min(100.0, float(percent)))
    try:
        _ctrl().SetMasterVolumeLevelScalar(percent / 100.0, None)
    except Exception:
        _reset_ctrl()
        raise


def get_volume():
    try:
        return _ctrl().GetMasterVolumeLevelScalar() * 100.0
    except Exception:
        _reset_ctrl()
        raise


def toggle_mute():
    try:
        c = _ctrl()
        c.SetMute(not c.GetMute(), None)
    except Exception:
        _reset_ctrl()
        raise


def is_muted():
    try:
        return bool(_ctrl().GetMute())
    except Exception:
        _reset_ctrl()
        raise
