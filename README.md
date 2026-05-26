# HandVol

Gesture-controlled Windows volume + media using a webcam and MediaPipe.
Point your finger to scrub volume, close your fist to mute, open your palm
to play/pause.

| Gesture | Action |
|---|---|
| ☝️ Point up | Hold and move up/down to scrub volume |
| ✊ Closed fist | Toggle mute |
| ✋ Open palm | Toggle play/pause |
| ✌️ Victory | Focus Spotify (launch it if not running) |

No model training, no cloud calls, no per-app permissions. Runs locally
off the pretrained MediaPipe Gesture Recognizer.

## Requirements

- Windows 10/11
- Python 3.11+
- A USB webcam (built for the NexiGo 1080p, works with anything DirectShow-compatible)

## Setup

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the MediaPipe gesture model bundle into `models/`:

```
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

Save it as `models/gesture_recognizer.task`.

## Run

```powershell
python handvol.pyw
```

HandVol installs a **system-tray icon** (bottom-right, in the hidden-icon
overflow on a fresh Windows install — drag it onto the taskbar to pin).

- **Left-click the tray icon** → toggle the OpenCV preview window on/off.
- **Right-click** → menu with *Show preview* (checked when visible) and *Quit*.
- **Esc** inside the preview window also hides it.
- To quit, use the tray's *Quit* item (or kill `pythonw.exe` from Task Manager).

By default it starts tray-only (no window). Pass `--show` to launch with
the preview already open.

### Useful flags

| Flag | What it does |
|---|---|
| `--sensitivity 80` | Volume points per full-frame vertical travel. Higher = faster scrub. |
| `--smoothing 0.3` | EMA factor on finger position. Lower = laggier, higher = jitterier. |
| `--cam 0` | Webcam index, if you have more than one. |
| `--debug` | Print frame-by-frame state and scrub values. |
| `--no-audio` | Skip all pycaw/media calls — overlay only, for tuning. |
| `--show` | Start with the preview window already open (otherwise tray-only). |

## Launch silently at startup

1. Press `Win+R`, type `shell:startup`, Enter.
2. Right-click → New → Shortcut → browse to `handvol_startup.vbs` in this repo.
3. Next login, HandVol runs hidden. Quit anytime with **Ctrl+Shift+Q**.

The `.vbs` shim invokes `pythonw.exe` so no console window flashes.
It prefers `.venv\Scripts\pythonw.exe` if present, otherwise falls back
to whatever `pythonw` is on PATH.

## Project layout

```
handvol/
├── capture.py    Webcam + MediaPipe LIVE_STREAM wiring
├── scrubber.py   Pure-logic volume scrubber (anchor + EMA)
├── state.py      IDLE / SCRUB / IDLE_COOLDOWN debouncer
├── audio.py      pycaw wrapper (volume + mute)
├── media.py      Media play/pause key
└── overlay.py    OpenCV drawing helpers
handvol.pyw       Entry point: argparse, loop, event dispatch
tests/
└── test_scrubber.py   Unit tests for scrubber + state machine
```

See [`CONTEXT.md`](CONTEXT.md) for the full design doc: gesture-mapping
rationale, scrub algorithm derivation, state-machine debounce thresholds,
and acceptance criteria.

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
