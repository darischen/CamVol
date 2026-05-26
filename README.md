# HandVol

Gesture-controlled Windows volume + media using a webcam and MediaPipe.
Pinch your thumb and index in an OK sign to scrub volume, close your fist
to mute, open your palm to play/pause, and use sideways thumbs to skip
tracks.

| Gesture | Action |
|---|---|
| 👌 OK sign | Hold and move up/down to scrub volume |
| ✊ Closed fist | Toggle mute |
| ✋ Open palm | Toggle play/pause |
| ✌️ Victory | Focus Spotify (launch it if not running) |
| 🤟 ILoveYou | Close Spotify |
| 👉 Thumb sideways → right | Next track (works for either hand) |
| 👈 Thumb sideways → left | Previous track (works for either hand) |

Direction is reported in your real-world frame: thumb tip toward your
right = next, toward your left = previous, regardless of which hand.

No model training, no cloud calls, no per-app permissions. Runs locally
off the pretrained MediaPipe Gesture Recognizer, with custom
landmark-based detection layered on top for the OK sign and sideways
thumbs.

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

### Installing dlib on Windows

`face_recognition` depends on `dlib`, which has no official Windows
wheel on PyPI. Install a community-built wheel for your Python version
**before** running `pip install -r requirements.txt`:

1. Download a matching wheel from https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases (or any trusted source).
2. `pip install path\to\dlib-19.xx.x-cpxx-cpxx-win_amd64.whl`
3. Then run `pip install -r requirements.txt`.

dlib runs on CPU only via this wheel. Face encoding takes ~30-80 ms per
call; the runtime loop throttles it to ~3 Hz so the main FPS is
unaffected.

Download the MediaPipe gesture model bundle into `models/`:

```
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

Save it as `models/gesture_recognizer.task`.

Also download the face landmarker model:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Save it as `models/face_landmarker.task`.

After first launch, run the face calibration once via the tray icon's
**Calibrate face...** menu item (or `python -m handvol.calibration`).
Gestures are blocked until calibration has been completed.

## Run

```powershell
python handvol.pyw
```

HandVol installs a **system-tray icon** (bottom-right, in the hidden-icon
overflow on a fresh Windows install — drag it onto the taskbar to pin).

- **Left-click the tray icon** → toggle the OpenCV preview window on/off.
- **Right-click** → menu with *Show preview* (checked when visible),
  *Pause* (releases the camera; useful before joining a virtual meeting),
  *Calibrate face...* (launches the face calibration flow), and *Quit*.
- **Esc** inside the preview window also hides it.
- To quit, use the tray's *Quit* item (or kill `pythonw.exe` from Task Manager).

By default it starts tray-only (no window). Pass `--show` to launch with
the preview already open.

### Useful flags

| Flag | What it does |
|---|---|
| `--sensitivity 80` | Volume points per full-frame vertical travel. Higher = faster scrub. |
| `--smoothing 0.3` | EMA factor on pinch position. Lower = laggier, higher = jitterier. |
| `--cam 0` | Webcam index, if you have more than one. |
| `--debug` | Print frame-by-frame state and scrub values. |
| `--no-audio` | Skip all pycaw/media calls — overlay only, for tuning. |
| `--show` | Start with the preview window already open (otherwise tray-only). |

## Face recognition (gesture lock)

HandVol gates gesture control behind face recognition: only the
calibrated user's face will unlock gestures. With no profile, or when
the user's face is not visible, gestures are ignored and the preview
overlay shows **LOCKED** (red) or **NO PROFILE** (gray) in the top
right. When you are recognized, it shows **UNLOCKED** (green).

**Calibrate via the tray menu** — right-click the HandVol icon and
choose **Calibrate face...**, or run `python -m handvol.calibration`
from the command line. You will be walked through ~20 short poses
(looking up/down/left/right, diagonals, profile views, two close/far
variations). Hold each pose at a neutral resting expression for ~2
seconds. Headphones are fine for most poses; you will be prompted to
briefly remove over-ear headphones for the two profile captures so the
ear/jawline is visible.

The face profile is stored at `data/face_profile.npz` and is **never**
committed to git (the `data/` directory is gitignored). Face embeddings
are biometric data — keep the file local.

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

See [`CONTEXT.md`](CONTEXT.md) for the full design doc: gesture-mapping
rationale, scrub algorithm derivation, state-machine debounce thresholds,
and acceptance criteria.

### Gesture detection internals

MediaPipe's built-in classifier handles `Closed_Fist`, `Open_Palm`,
`Victory`, and `ILoveYou` directly. The OK sign and the four sideways
thumb labels (`left_hand_thumb_left`, `left_hand_thumb_right`,
`right_hand_thumb_left`, `right_hand_thumb_right`) are detected from the
21 hand landmarks in `capture.py`:

- **OK sign:** thumb tip touches index tip; middle, ring, pinky extended.
- **Sideways thumb:** thumb extended horizontally with the other four
  fingers curled into a fist. Direction comes from the thumb-tip x
  relative to the thumb base; hand identity comes from MediaPipe's
  handedness (with the convention inverted because the frame is
  pre-mirrored for selfie view).

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
