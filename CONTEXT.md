# HandVol: Gesture-Controlled Volume

## What This Is
A Python app that uses my NexiGo 1080p30 webcam + MediaPipe to control
Windows system volume and media playback via hand gestures.

## Hardware Context
- Webcam: NexiGo N60 1080p @ 30fps, USB
- GPU: RTX 3060 Ti (overkill, MediaPipe runs fine on CPU)
- OS: Windows
- Python: 3.11+ preferred

## Final Gesture Mapping
| Gesture | MediaPipe label | Action |
|---|---|---|
| Point up | Pointing_Up | Enter scrub mode for volume |
| Fist | Closed_Fist | Toggle system mute |
| Open palm | Open_Palm | Toggle media play/pause |

These three are all in MediaPipe's pretrained Gesture Recognizer. No
custom model training needed. Do NOT propose collecting data or
training an MLP. We already decided against it.

## Stack (Locked In)
- `mediapipe` (Tasks API, GestureRecognizer in LIVE_STREAM mode)
- `opencv-python` for capture + overlay
- `pycaw` for Windows audio
- `keyboard` library for media key injection
- `numpy` for math

Use the MediaPipe Tasks API (`mediapipe.tasks.python.vision`), NOT the
legacy `mp.solutions.hands` API. Download the model bundle from:
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

## Capture Settings
- Resolution: downsample to 640x480 before inference (1080p is wasted
  bandwidth for landmark detection)
- Backend: `cv2.CAP_DSHOW` on Windows (NexiGo behaves better with
  DirectShow than MSMF)
- Mirror the frame horizontally so the overlay matches user expectations

## Scrub Algorithm (Scheme 2: Relative Offset)
On entering scrub state, anchor both the index tip Y and current
system volume. Then map vertical delta to volume delta.

```python
class VolumeScrubber:
    def __init__(self, sensitivity=80, smoothing=0.3):
        self.anchor_y = None
        self.anchor_vol = None
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        self.smoothed_y = None

    def enter(self, tip_y, current_vol):
        self.anchor_y = tip_y
        self.anchor_vol = current_vol
        self.smoothed_y = tip_y

    def update(self, tip_y):
        self.smoothed_y = (self.smoothing * tip_y +
                          (1 - self.smoothing) * self.smoothed_y)
        delta = self.anchor_y - self.smoothed_y
        new_vol = self.anchor_vol + self.sensitivity * delta
        return max(0, min(100, new_vol))

    def exit(self):
        self.anchor_y = None
```

Sensitivity = 80 means full vertical frame travel covers 80 volume
points. Smoothing is EMA on the index tip Y to kill landmark jitter.

`tip_y` is normalized to [0, 1] using frame height. Use landmark index
8 from `result.hand_landmarks[0]`.

## State Machine

States: IDLE, SCRUB, IDLE_COOLDOWN

- IDLE -> SCRUB: 5 consecutive frames of Pointing_Up. Call scrubber.enter().
- SCRUB -> SCRUB: while Pointing_Up, call scrubber.update() and apply volume.
- SCRUB -> IDLE: 3 consecutive frames of non-Pointing_Up. Call scrubber.exit().
- IDLE -> toggle mute: 5 consecutive frames of Closed_Fist (edge-triggered),
  then IDLE_COOLDOWN for 10 frames.
- IDLE -> toggle play/pause: 5 consecutive frames of Open_Palm (edge-triggered),
  then IDLE_COOLDOWN for 10 frames.
- IDLE_COOLDOWN -> IDLE: 10 frames of neutral/no-gesture.

Asymmetric debounce (5 enter, 3 exit) keeps scrub responsive while
preventing flicker. Edge-triggering on fist/palm prevents toggle spam
while gesture is held.

## OS Control Layer

### Volume (pycaw)
```python
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

def set_volume(percent):
    volume_ctrl.SetMasterVolumeLevelScalar(percent / 100.0, None)

def get_volume():
    return volume_ctrl.GetMasterVolumeLevelScalar() * 100

def toggle_mute():
    volume_ctrl.SetMute(not volume_ctrl.GetMute(), None)
```

### Media Keys (keyboard library)
```python
import keyboard
keyboard.send('play/pause media')
```

Note: the `keyboard` package needs admin on some Windows configs. If
that's a problem, fall back to `pyautogui.press('playpause')`.

## Visual Overlay (Required, not optional)
On the OpenCV display window:
- Current state name (IDLE / SCRUB / COOLDOWN) top-left
- Current detected gesture top-left, below state
- Current system volume number top-right
- In SCRUB state:
  - Horizontal cyan line across frame at anchor_y
  - Dot on index tip at current position
  - Vertical line connecting them
- Detected hand landmarks drawn faintly when any hand is present

Without this overlay, tuning sensitivity is guessing. Build it
into Phase 1, not as polish later.

## Project Structure
```
handvol/
├── CONTEXT.md          (this file)
├── requirements.txt
├── models/
│   └── gesture_recognizer.task   (downloaded once)
├── handvol/
│   ├── __init__.py
│   ├── capture.py      (webcam + MediaPipe wiring)
│   ├── scrubber.py     (VolumeScrubber class)
│   ├── state.py        (state machine + debouncer)
│   ├── audio.py        (pycaw wrapper)
│   ├── media.py        (media key wrapper)
│   ├── overlay.py      (OpenCV drawing functions)
│   └── main.py         (entry point, glues everything)
└── tests/
    └── test_scrubber.py
```

## Build Order (Strict)

### Phase 1: Capture + Recognizer + Overlay (no actions yet)
- Webcam opens at 640x480 with DSHOW backend
- GestureRecognizer in LIVE_STREAM mode with async callback
- Display window shows live feed, detected gesture name, landmarks
- Verify >= 25fps sustained
- DO NOT call any audio APIs yet

### Phase 2: State Machine
- Implement debouncer with frame counts above
- Print state transitions to console
- No audio calls yet, just verify transitions trigger correctly
  by making each gesture

### Phase 3: Audio Integration
- Wire `Closed_Fist` edge to `toggle_mute()`
- Wire `Open_Palm` edge to media play/pause
- Wire `Pointing_Up` to VolumeScrubber + `set_volume()`
- Add scrub-mode overlay (anchor line, tip dot)

### Phase 4: Tuning
- Expose sensitivity and smoothing as CLI flags or a JSON config
- Add a `--debug` flag that prints frame-by-frame state + values

### Phase 5 (Optional): System tray
- pystray icon to enable/disable without killing process
- Global hotkey (Ctrl+Shift+H) to toggle active state

## Known Edge Cases to Handle From Day One
1. Hand leaves frame mid-scrub: 3-frame debounce exits SCRUB cleanly,
   volume stays where it was last set.
2. Gesture flicker between Pointing_Up and Victory: 5-frame entry debounce
   prevents accidental scrub start.
3. MediaPipe returns empty `result.gestures` when confidence is low:
   treat as "no gesture this frame" for the debouncer.
4. EMA smoothing has lag on intentional fast movements: 0.3 factor is the
   tested sweet spot. Lower = more lag, higher = more jitter.

## Do Not Suggest
- Training a custom classifier. Pretrained handles all three gestures.
- Using a deep learning model for the scrub logic. It's arithmetic.
- Using absolute Y -> volume mapping. Already rejected. Causes snap-to.
- Using velocity-zone (joystick) scrubbing. Already rejected. Harder to
  land on specific values.
- Adding more gestures right now. Ship three first.
- Tracking only the index tip without the rest of the landmarks. The
  classifier needs all 21 to fire Pointing_Up.

## Acceptance Criteria
- Cold start to first volume change: < 5 seconds
- End-to-end latency (gesture -> volume change): < 150ms
- CPU usage: < 15% on a modern desktop
- Zero crashes over a 30-minute session

## First Task When You Start
Read this entire doc back to me in your own words to confirm
understanding, then propose the contents of `requirements.txt` and the
Phase 1 file list before writing any code.