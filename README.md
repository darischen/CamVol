# HandVol

Gesture-controlled Windows volume + media using a webcam and MediaPipe.
Pinch your thumb and index in an OK sign to scrub volume, close your fist
to mute, open your palm to play/pause, sideways thumbs to skip tracks, and
number gestures (1-10) formed with two hands. Hold a middle finger for 5 seconds
to restart your PC, or double middle finger to shut it down.

| Gesture | Action |
|---|---|
| 👌 OK sign | Hold and move up/down to scrub volume |
| ✊ Closed fist | Toggle mute |
| ✋ Open palm | Toggle play/pause |
| ✌️ Victory | Focus Spotify (launch it if not running) |
| 🤟 ILoveYou | Close Spotify |
| 👉 Thumb sideways → right | Next track (works for either hand) |
| 👈 Thumb sideways → left | Previous track (works for either hand) |
| 🤘 Hang loose sign | Pause app & release camera |
| **Number gestures (two hands)** | **Display recognized number 1-10** |
| Fist + Pointer | Number 1 — Focus first Chrome window (Win+1; launch Chrome if not running) |
| Fist + Victory | Number 2 — Cycle to second Chrome window (Win+1 twice) |
| Fist + 3 fingers | Number 3 — Focus Discord (launch it if not running) |
| Fist + 4 fingers | Number 4 — Focus VS Code (launch it if not running) |
| Fist + Open palm | Number 5 — Open Task Manager (Ctrl+Shift+Esc) |
| Open palm + Pointer | Number 6 — Voice search (focus Chrome, Ctrl+T, dictate, auto-Enter after 1s silence) |
| Open palm + Victory | Number 7 |
| Open palm + 3 fingers | Number 8 |
| Open palm + 4 fingers | Number 9 — Lock Gesture |
| Open palm + Open palm | Number 10 — Close active window (Alt+F4) |
| 🖕 Single middle finger (3s hold) | **Restart Windows** |
| 🖕🖕 Double middle finger (3s hold) | **Shut down Windows** |

Direction is reported in your real-world frame: thumb tip toward your
right = next, toward your left = previous, regardless of which hand.

No model training, no cloud calls, no per-app permissions. Runs locally
off the pretrained MediaPipe Gesture Recognizer, with custom
landmark-based detection layered on top for the OK sign and sideways
thumbs.

## Extra Features

**Lock State:** When the lock icon displays red in the overlay, gesture
execution is blocked to prevent accidental actions. Gestures are still
recognized, allowing number_9 (nine fingers) to toggle the lock
even when active. Use this before focusing on other tasks or when you
don't want HandVol responding to your hands.

**Camera Release:** The hang loose sign (shaka) pauses the app and releases
the camera, or use the *Pause* option in the tray menu. Useful before
joining video calls or when another app needs camera access. Resume by
toggling the pause state again.

**Voice Search:** Form `number_6` (open palm + pointer) to focus Chrome,
open a new tab, and start dictating. Your speech is transcribed locally
with `faster-whisper` (`base.en`, int8, CPU) — no cloud calls. The tray
icon turns into a red microphone and HandVol auto-locks gestures while
recording. After 1 second of silence, the transcript is typed into the
URL bar and Enter fires automatically. If no speech is detected within
~5 seconds, the trigger times out cleanly with no typing.

First invocation downloads the `base.en` model (~140 MB) into the
HuggingFace cache.

## Requirements

- Windows 10/11
- Python 3.11+
- A USB webcam (built for the NexiGo 1080p, works with anything DirectShow-compatible)

## Setup

Install dependencies directly into your system Python (no virtualenv):

```powershell
# from the repo root
pip install -r requirements.txt
```

## Run

```powershell
python handvol.pyw
```

HandVol installs a **system-tray icon** (bottom-right, in the hidden-icon
overflow on a fresh Windows install — drag it onto the taskbar to pin).

- **Left-click the tray icon** → toggle the OpenCV preview window on/off.
- **Right-click** → menu with *Show preview* (checked when visible),
  *Pause* (releases the camera; useful before joining a virtual meeting),
  and *Quit*.
- **Esc** inside the preview window also hides it.
- To quit, use the tray's *Quit* item (or kill `pythonw.exe` from Task Manager).

By default it starts tray-only (no window). Pass `--show` to launch with
the preview already open.

**Hold gesture feedback:** When you hold a middle finger (or double middle finger),
the preview overlay displays a red timer showing elapsed time and the action it
will trigger (e.g., `RESTART 2.3s / 3.0s`). The action fires once you hold for
the full 3.0 seconds.

### Useful flags

| Flag | What it does |
|---|---|
| `--sensitivity 80` | Volume points per full-frame vertical travel. Higher = faster scrub. |
| `--smoothing 0.3` | EMA factor on pinch position. Lower = laggier, higher = jitterier. |
| `--cam 0` | Webcam index, if you have more than one. |
| `--debug` | Print frame-by-frame state and scrub values. |
| `--no-audio` | Skip all pycaw/media calls — overlay only, for tuning. |
| `--show` | Start with the preview window already open (otherwise tray-only). |

## Launch silently at startup

1. Press `Win+R`, type `shell:startup`, Enter.
2. Right-click → New → Shortcut → browse to `handvol_startup.vbs` in this repo.
3. Next login, HandVol runs hidden. Quit anytime via the tray icon.

The `.vbs` shim invokes `pythonw.exe` so no console window flashes.

## Project layout

```
handvol/
├── capture.py        Webcam + MediaPipe LIVE_STREAM + custom OK/side-thumb detection
├── scrubber.py       Pure-logic volume scrubber (anchor + EMA)
├── state.py          IDLE / SCRUB / IDLE_COOLDOWN debouncer
├── audio.py          pycaw wrapper (volume + mute)
├── media.py          Media play/pause and track-skip keys
├── spotify.py        Focus/launch/close Spotify via Win32
├── discord.py        Focus/launch Discord via Win32 + Squirrel updater
├── vscode.py         Focus/launch VS Code via Win32 + known install paths
├── taskbar.py        Synthesize Win+N taskbar shortcuts (used for Chrome)
├── overlay.py        OpenCV drawing helpers
├── face_detect.py    MediaPipe Face Landmarker wrapper + landmark-to-embedding helper
├── face_profile.py   On-disk face identity profile (cosine similarity matching)
└── calibration.py    Standalone face calibration UI (~20 pose captures)
handvol.pyw           Entry point: argparse, tray icon, capture loop, event dispatch
tests/
├── test_scrubber.py        Unit tests for scrubber + state machine
├── test_face_embedding.py  Unit tests for landmark-to-embedding helper
└── test_face_profile.py    Unit tests for FaceProfile save/load + matching
```

### Gesture detection internals

MediaPipe's built-in classifier handles `Closed_Fist`, `Open_Palm`,
`Victory`, and `ILoveYou` directly. Custom landmark-based detection in
`capture.py` covers:

- **OK sign:** thumb tip touches index tip; middle, ring, pinky extended.
- **Sideways thumb:** thumb extended horizontally with the other four
  fingers curled into a fist. Direction comes from the thumb-tip x
  relative to the thumb base; hand identity comes from MediaPipe's
  handedness (with the convention inverted because the frame is
  pre-mirrored for selfie view).
- **Hang loose sign:** thumb and pinky extended, other three fingers
  curled into a fist. Triggers pause/resume and camera release.

The lock icon in the overlay indicates the current lock state (unlocked
by default). Lock state is toggled via number_9 (double middle finger)
and blocks all gesture-triggered actions while still recognizing gestures
in the frame.

Tuning constants for the landmark checks live at the top of
`handvol/capture.py`.

## Testing

```powershell
python -m pytest tests/ -q
```

The scrubber and state machine are pure logic and tested without a
webcam.

## Troubleshooting

- **`Could not open camera index 0`** — another app has the camera, or
  Windows Camera privacy settings are blocking desktop apps. Try
  `--cam 1`.
- **`AudioDevice has no attribute Activate`** — pycaw version mismatch.
  The code handles both old and new pycaw via `_dev` fallback; make
  sure you're on the version pinned in `requirements.txt`.
- **Media key does nothing** — the `keyboard` library needs admin on
  some Windows configs. Run from an elevated terminal, or `pyautogui`
  will be tried as fallback if installed.
- **Tray icon missing** — Windows hides new tray icons in the overflow
  by default. Click the `^` chevron in the taskbar and drag the HandVol
  icon onto the visible area to pin it.
- **Side-thumb skip/prev firing the wrong direction** — extremely
  unlikely, but if MediaPipe's handedness convention differs on your
  build, swap the `hand_prefix` mapping in
  `handvol/capture.py::_detect_side_thumb`.
