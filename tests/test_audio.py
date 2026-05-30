"""
Tests for handvol.audio cache-invalidation on default device change.

Strategy: monkeypatch AudioUtilities.GetSpeakers so each call returns a
distinct sentinel object — that lets us count how many times _ctrl() actually
rebuilds the interface.  We also monkeypatch the enumerator registration so
the test never touches real COM.
"""

import types
import pytest
import handvol.audio as audio_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeInterface:
    """Minimal stand-in for IAudioEndpointVolume pointer."""

    def __init__(self, identity):
        self._identity = identity

    def GetMasterVolumeLevelScalar(self):
        return 0.5

    def SetMasterVolumeLevelScalar(self, val, ctx):
        pass

    def GetMute(self):
        return False

    def SetMute(self, val, ctx):
        pass


class _FakeDevice:
    """Returned by the patched GetSpeakers; Activate produces a FakeInterface."""

    def __init__(self, identity):
        self._identity = identity

    def Activate(self, iid, ctx, params):
        # Return a _FakeInterface; audio._ctrl() will cast() it, which we patch
        # to be identity, so _volume_ctrl ends up as a _FakeInterface.
        return _FakeInterface(self._identity)

    # Give it a _dev attribute so audio._ctrl()'s getattr path still works
    @property
    def _dev(self):
        return self


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_audio_module():
    """Reset all module globals in handvol.audio before and after each test."""
    audio_mod._volume_ctrl = None
    audio_mod._enumerator = None
    audio_mod._notification_client = None
    audio_mod._watcher_attempted = False
    yield
    audio_mod._volume_ctrl = None
    audio_mod._enumerator = None
    audio_mod._notification_client = None
    audio_mod._watcher_attempted = False


@pytest.fixture
def patched_audio(monkeypatch):
    """
    Patch AudioUtilities.GetSpeakers + comtypes cast so _ctrl() builds
    FakeInterface sentinels without hitting real COM.
    Also patch the enumerator so registration never touches COM.

    Relies on the autouse reset_audio_module fixture to isolate module
    globals (including _watcher_attempted) before/after each test.
    """
    call_count = {"n": 0}

    def fake_get_speakers():
        call_count["n"] += 1
        return _FakeDevice(identity=call_count["n"])

    # Patch GetSpeakers on the pycaw module that audio.py already imported
    import pycaw.pycaw as pycaw_mod
    monkeypatch.setattr(pycaw_mod.AudioUtilities, "GetSpeakers", staticmethod(fake_get_speakers))

    # Patch cast so it returns the device object itself (our FakeInterface)
    import handvol.audio as am
    monkeypatch.setattr(am, "cast", lambda obj, ptr_type: obj)

    # Patch GetDeviceEnumerator so we never touch real COM and can capture
    # the client that gets registered
    registered = {"client": None}

    class _FakeEnumerator:
        def RegisterEndpointNotificationCallback(self, client):
            registered["client"] = client

        def UnregisterEndpointNotificationCallback(self, client):
            pass

    monkeypatch.setattr(
        pycaw_mod.AudioUtilities,
        "GetDeviceEnumerator",
        staticmethod(lambda: _FakeEnumerator()),
    )

    return call_count, registered


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_volume_builds_interface_once(patched_audio):
    """Repeated get_volume() calls reuse the cached interface."""
    call_count, _ = patched_audio
    audio_mod.get_volume()
    audio_mod.get_volume()
    audio_mod.get_volume()
    assert call_count["n"] == 1, "Interface should be built only once for repeated calls"


def test_render_device_change_invalidates_cache(patched_audio):
    """
    After the first get_volume(), simulating a render device change via the
    registered MMNotificationClient must cause the NEXT get_volume() to rebuild
    the interface (i.e. call GetSpeakers again).
    """
    call_count, registered = patched_audio

    # Build the cache with the first call
    audio_mod.get_volume()
    assert call_count["n"] == 1

    # Simulate Windows firing OnDefaultDeviceChanged for the render direction.
    # The notification client should have been registered during _ctrl() init.
    client = registered["client"]
    assert client is not None, (
        "audio module must register an MMNotificationClient on first _ctrl() call"
    )

    # flow_id=0 → "eRender" per pycaw MMNotificationClient.DataFlow
    client.on_default_device_changed(
        flow="eRender", flow_id=0, role="eConsole", role_id=0,
        default_device_id="{fake-new-device-id}"
    )

    # Next call must rebuild (GetSpeakers called a second time)
    audio_mod.get_volume()
    assert call_count["n"] == 2, (
        "Cache must be invalidated after a render default-device change; "
        "GetSpeakers should be called again"
    )


def test_capture_device_change_does_not_invalidate_cache(patched_audio):
    """
    A microphone (eCapture) default-device change must NOT churn the render cache.
    """
    call_count, registered = patched_audio

    audio_mod.get_volume()
    assert call_count["n"] == 1

    client = registered["client"]
    assert client is not None

    # flow_id=1 → "eCapture"
    client.on_default_device_changed(
        flow="eCapture", flow_id=1, role="eConsole", role_id=0,
        default_device_id="{fake-mic-id}"
    )

    audio_mod.get_volume()
    assert call_count["n"] == 1, (
        "Cache must NOT be invalidated for eCapture device changes"
    )


def test_registration_failure_does_not_break_volume_control(monkeypatch):
    """
    If RegisterEndpointNotificationCallback raises, volume control must still work.
    """
    import pycaw.pycaw as pycaw_mod

    call_count = {"n": 0}

    def fake_get_speakers():
        call_count["n"] += 1
        return _FakeDevice(identity=call_count["n"])

    monkeypatch.setattr(pycaw_mod.AudioUtilities, "GetSpeakers", staticmethod(fake_get_speakers))
    import handvol.audio as am
    monkeypatch.setattr(am, "cast", lambda obj, ptr_type: obj)

    class _BrokenEnumerator:
        def RegisterEndpointNotificationCallback(self, client):
            raise OSError("COM not initialized")

    monkeypatch.setattr(
        pycaw_mod.AudioUtilities,
        "GetDeviceEnumerator",
        staticmethod(lambda: _BrokenEnumerator()),
    )

    # Module globals are reset by the autouse reset_audio_module fixture, so
    # this test starts from a clean slate without any manual setup.

    # Must not raise; basic volume ops must still work
    vol = am.get_volume()
    assert isinstance(vol, float)
    am.set_volume(50)
    assert call_count["n"] >= 1
    # (no manual teardown needed — autouse reset_audio_module handles it)


def test_failed_registration_is_not_retried(monkeypatch):
    """
    When registration fails permanently, _ensure_watcher() must attempt
    GetDeviceEnumerator exactly once — not on every _ctrl() call.

    Regression test for the pathological-retry bug: if _watcher_attempted
    is not set on failure, every get_volume() / set_volume() call would
    re-run expensive cross-process COM setup.
    """
    import pycaw.pycaw as pycaw_mod
    import handvol.audio as am

    call_count = {"n": 0}

    def fake_get_speakers():
        call_count["n"] += 1
        return _FakeDevice(identity=call_count["n"])

    monkeypatch.setattr(pycaw_mod.AudioUtilities, "GetSpeakers", staticmethod(fake_get_speakers))
    monkeypatch.setattr(am, "cast", lambda obj, ptr_type: obj)

    enum_call_count = {"n": 0}

    class _BrokenEnumerator:
        def RegisterEndpointNotificationCallback(self, client):
            raise OSError("COM not initialized")

    def fake_get_enumerator():
        enum_call_count["n"] += 1
        return _BrokenEnumerator()

    monkeypatch.setattr(
        pycaw_mod.AudioUtilities,
        "GetDeviceEnumerator",
        staticmethod(fake_get_enumerator),
    )

    # Drive multiple volume calls, resetting the volume interface each time
    # to force _ctrl() (and therefore _ensure_watcher()) to run on every call.
    # Registration fails on the first attempt; without the sentinel fix, every
    # subsequent call would retry GetDeviceEnumerator unnecessarily.
    for _ in range(5):
        am._volume_ctrl = None  # force _ctrl() to re-enter on the next call
        am.get_volume()

    assert enum_call_count["n"] == 1, (
        "GetDeviceEnumerator must be called at most once even when registration "
        f"permanently fails (was called {enum_call_count['n']} times)"
    )
