# Voice Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `number_6` gesture so it focuses Chrome, opens a new tab, captures voice via the mic, transcribes it offline with faster-whisper, types the result into the URL bar, and presses Enter after 1 second of silence.

**Architecture:** A new background thread (`VoiceSearch`) owns the microphone, runs a `webrtcvad`-based silence detector, calls `faster-whisper` for transcription, and types via pyautogui. The main `capture_loop` dispatches the new `VOICE_SEARCH` event by reusing the existing lock mechanism (same code path as `NUMBER_9`) and the Chrome-focus path (same as `NUMBER_1`). A shared `voice_state` dict communicates completion back to the main loop, where the lock is restored to its pre-trigger value and the tray glyph swaps back from the mic icon.

**Tech Stack:** Python 3.11+, `faster-whisper` (`base.en`, `int8`, CPU), `sounddevice` (mic capture), `webrtcvad` (silence detection), pyautogui (typing + Ctrl+T).

---

## File Structure

| File | Role |
|---|---|
| `handvol/voice_search.py` | **New.** `SilenceDetector` (pure logic, tested) + `VoiceSearch` (mic/Whisper/typing orchestrator). |
| `handvol/shortcuts.py` | Add `open_new_tab()` (Ctrl+T with the same modifier-warmup pattern). |
| `handvol/state.py` | Add `NUMBER_6` constant, `VOICE_SEARCH` event, counter + dispatcher arm. |
| `handvol.pyw` | Add `make_mic_image()`, load Whisper model at startup, wire `VOICE_SEARCH` dispatch, poll `voice_state` for completion. |
| `requirements.txt` | Add `faster-whisper`, `sounddevice`, `webrtcvad`. |
| `tests/test_voice_search.py` | **New.** Unit tests for `SilenceDetector` + state machine `NUMBER_6` → `VOICE_SEARCH` transition. |
| `README.md` | Fill the `number_6` row; add a "Voice search" feature blurb. |

---

## Task 1: Add NUMBER_6 constant and VOICE_SEARCH event

**Files:**
- Modify: `handvol/state.py`
- Modify: `tests/test_scrubber.py` (add new tests in same file — matches existing layout)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scrubber.py`:

```python
from handvol.state import NUMBER_6, TOGGLE_FRAMES


def test_state_machine_number_6_fires_voice_search():
    sm = GestureStateMachine()
    events = [sm.step(NUMBER_6) for _ in range(TOGGLE_FRAMES)]
    assert events[-1] is Event.VOICE_SEARCH
    assert all(e is Event.NONE for e in events[:-1])


def test_state_machine_number_6_resets_after_cooldown():
    sm = GestureStateMachine()
    for _ in range(TOGGLE_FRAMES):
        sm.step(NUMBER_6)
    # After firing once, machine enters IDLE_COOLDOWN; should not refire immediately.
    next_events = [sm.step(NUMBER_6) for _ in range(TOGGLE_FRAMES)]
    assert Event.VOICE_SEARCH not in next_events
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scrubber.py::test_state_machine_number_6_fires_voice_search -v`
Expected: FAIL with `ImportError: cannot import name 'NUMBER_6'` (or `AttributeError: Event.VOICE_SEARCH`).

- [ ] **Step 3: Add constant, event, counter, dispatcher**

In `handvol/state.py`:

After the existing `NUMBER_5 = "Number_5"` line (around line 64), add:

```python
NUMBER_6 = "Number_6"
```

In the `Event` enum (around line 11), add a new member after `TOGGLE_LOCK`:

```python
    VOICE_SEARCH = "voice_search"
```

In `__init__` (after `self._number_5_count = 0`), add:

```python
        self._number_6_count = 0
```

In `_reset_counters` (after `self._number_5_count = 0`), add:

```python
        self._number_6_count = 0
```

In `_bump` (after the `_number_5_count` assignment), add:

```python
        self._number_6_count = self._number_6_count + 1 if gesture == NUMBER_6 else 0
```

In `_bump`'s `if is_skip or is_prev or gesture in (...)` tuple (line ~159), add `NUMBER_6` to the list:

```python
            NUMBER_1, NUMBER_2, NUMBER_3, NUMBER_4, NUMBER_5, NUMBER_6, NUMBER_9, NUMBER_10,
```

In `step`'s IDLE branch, after the `_number_5_count` block (around line 252), add:

```python
            if self._number_6_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.VOICE_SEARCH
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scrubber.py -v`
Expected: PASS for both new tests; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add handvol/state.py tests/test_scrubber.py
git commit -m "Add NUMBER_6 gesture and VOICE_SEARCH event"
```

---

## Task 2: Add open_new_tab() shortcut helper

**Files:**
- Modify: `handvol/shortcuts.py`

- [ ] **Step 1: Add the helper**

In `handvol/shortcuts.py`, after `open_task_manager()`:

```python
def open_new_tab():
    """Send Ctrl+T to open a new tab in the active window. Returns 'ok' or 'failed'.

    Mirrors close_window()'s modifier-warmup + try/finally pattern so the
    Ctrl modifier is always released, even on exception.
    """
    try:
        pyautogui.keyDown("ctrl")
        try:
            time.sleep(MODIFIER_WARMUP)
            pyautogui.press("t")
        finally:
            time.sleep(INTER_KEY_DELAY)
            pyautogui.keyUp("ctrl")
        return "ok"
    except Exception:
        return "failed"
```

Update the module docstring's first line (line 3) from:

```
Alt+F4 for close window, Ctrl+Shift+Esc for Task Manager.
```

to:

```
Alt+F4 for close window, Ctrl+Shift+Esc for Task Manager, Ctrl+T for new tab.
```

- [ ] **Step 2: Sanity-import**

Run: `python -c "from handvol.shortcuts import open_new_tab; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add handvol/shortcuts.py
git commit -m "Add open_new_tab() shortcut helper"
```

---

## Task 3: Build SilenceDetector (pure-logic VAD state machine)

**Files:**
- Create: `handvol/voice_search.py`
- Create: `tests/test_voice_search.py`

Design: `SilenceDetector` is a pure-logic class that consumes a stream of `is_speech: bool` flags (one per 30 ms audio frame) and reports the current phase. It owns no audio — that's the orchestrator's job. This makes it trivially testable.

States:
- `WAITING_FOR_SPEECH` — initial. If `initial_silence_frames` go by without any speech, transition to `TIMEOUT`.
- `IN_SPEECH` — entered after `speech_start_frames` consecutive speech frames. Track a rolling silence counter; when it hits `silence_end_frames`, transition to `DONE`.
- `DONE` — recording complete, time to transcribe.
- `TIMEOUT` — no speech detected; abort without transcribing.

At 30 ms/frame:
- `speech_start_frames = 10` (~300 ms debounce against background bursts)
- `silence_end_frames = 33` (~1.0 s of silence)
- `initial_silence_frames = 166` (~5.0 s timeout)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice_search.py`:

```python
from handvol.voice_search import SilenceDetector, Phase


def test_starts_in_waiting_phase():
    d = SilenceDetector()
    assert d.phase is Phase.WAITING_FOR_SPEECH


def test_timeout_when_no_speech():
    d = SilenceDetector(initial_silence_frames=5)
    for _ in range(5):
        d.feed(is_speech=False)
    assert d.phase is Phase.TIMEOUT


def test_enters_speech_after_debounce():
    d = SilenceDetector(speech_start_frames=3)
    d.feed(is_speech=True)
    d.feed(is_speech=True)
    assert d.phase is Phase.WAITING_FOR_SPEECH  # not yet
    d.feed(is_speech=True)
    assert d.phase is Phase.IN_SPEECH


def test_short_burst_does_not_trigger_speech():
    d = SilenceDetector(speech_start_frames=3)
    d.feed(is_speech=True)
    d.feed(is_speech=False)  # gap resets the debounce
    d.feed(is_speech=True)
    d.feed(is_speech=True)
    # only 2 consecutive speech frames so far → still waiting
    assert d.phase is Phase.WAITING_FOR_SPEECH


def test_done_after_silence_in_speech():
    d = SilenceDetector(speech_start_frames=2, silence_end_frames=4)
    # Enter speech
    d.feed(is_speech=True)
    d.feed(is_speech=True)
    assert d.phase is Phase.IN_SPEECH
    # Brief silence (not enough)
    for _ in range(3):
        d.feed(is_speech=False)
    assert d.phase is Phase.IN_SPEECH
    # Total of 4 consecutive silence frames → DONE
    d.feed(is_speech=False)
    assert d.phase is Phase.DONE


def test_speech_during_silence_resets_silence_counter():
    d = SilenceDetector(speech_start_frames=1, silence_end_frames=4)
    d.feed(is_speech=True)
    assert d.phase is Phase.IN_SPEECH
    d.feed(is_speech=False)
    d.feed(is_speech=False)
    d.feed(is_speech=True)  # resets
    d.feed(is_speech=False)
    d.feed(is_speech=False)
    d.feed(is_speech=False)
    assert d.phase is Phase.IN_SPEECH  # only 3 silence frames since last speech


def test_feed_after_done_is_noop():
    d = SilenceDetector(speech_start_frames=1, silence_end_frames=1)
    d.feed(is_speech=True)
    d.feed(is_speech=False)
    assert d.phase is Phase.DONE
    d.feed(is_speech=True)
    assert d.phase is Phase.DONE  # terminal
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_voice_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'handvol.voice_search'`.

- [ ] **Step 3: Implement SilenceDetector**

Create `handvol/voice_search.py`:

```python
"""Voice search: capture mic audio, transcribe with faster-whisper offline,
type the result into the focused window, press Enter after a configurable
silence threshold.

This module exposes two units:

- ``SilenceDetector``: pure-logic VAD state machine. Consumes a stream of
  ``is_speech`` booleans (one per audio frame) and reports the current phase.
  Tested in isolation.

- ``VoiceSearch``: orchestrator that owns the microphone, runs ``webrtcvad``
  against incoming frames, feeds them to ``SilenceDetector``, then transcribes
  the captured buffer with faster-whisper and types the result via pyautogui.
  Integration-only; not unit-tested.
"""
from enum import Enum


class Phase(str, Enum):
    WAITING_FOR_SPEECH = "waiting_for_speech"
    IN_SPEECH = "in_speech"
    DONE = "done"
    TIMEOUT = "timeout"


# Defaults assume 30 ms frames (webrtcvad's native frame size at 16 kHz).
# 10 frames * 30 ms = 300 ms speech debounce
# 33 frames * 30 ms ≈ 1.0 s end-of-utterance silence
# 166 frames * 30 ms ≈ 5.0 s no-speech timeout
DEFAULT_SPEECH_START_FRAMES = 10
DEFAULT_SILENCE_END_FRAMES = 33
DEFAULT_INITIAL_SILENCE_FRAMES = 166


class SilenceDetector:
    """Pure-logic VAD phase tracker. Feed one is_speech bool per audio frame.

    Phase transitions:
        WAITING_FOR_SPEECH → IN_SPEECH  after speech_start_frames consecutive speech
        WAITING_FOR_SPEECH → TIMEOUT    after initial_silence_frames with no speech
        IN_SPEECH          → DONE       after silence_end_frames consecutive silence

    DONE and TIMEOUT are terminal.
    """

    def __init__(
        self,
        speech_start_frames=DEFAULT_SPEECH_START_FRAMES,
        silence_end_frames=DEFAULT_SILENCE_END_FRAMES,
        initial_silence_frames=DEFAULT_INITIAL_SILENCE_FRAMES,
    ):
        self.speech_start_frames = speech_start_frames
        self.silence_end_frames = silence_end_frames
        self.initial_silence_frames = initial_silence_frames
        self.phase = Phase.WAITING_FOR_SPEECH
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._total_frames_in_waiting = 0

    def feed(self, is_speech):
        if self.phase in (Phase.DONE, Phase.TIMEOUT):
            return

        if self.phase is Phase.WAITING_FOR_SPEECH:
            self._total_frames_in_waiting += 1
            if is_speech:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self.speech_start_frames:
                    self.phase = Phase.IN_SPEECH
                    self._consecutive_silence = 0
            else:
                self._consecutive_speech = 0
                if self._total_frames_in_waiting >= self.initial_silence_frames:
                    self.phase = Phase.TIMEOUT
            return

        # IN_SPEECH
        if is_speech:
            self._consecutive_silence = 0
        else:
            self._consecutive_silence += 1
            if self._consecutive_silence >= self.silence_end_frames:
                self.phase = Phase.DONE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_voice_search.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add handvol/voice_search.py tests/test_voice_search.py
git commit -m "Add SilenceDetector pure-logic VAD state machine"
```

---

## Task 4: Build VoiceSearch orchestrator (mic + Whisper + typing)

**Files:**
- Modify: `handvol/voice_search.py`

- [ ] **Step 1: Append VoiceSearch class**

Add to `handvol/voice_search.py` (after `SilenceDetector`):

```python
import queue
import threading
import time

import numpy as np
import pyautogui
import sounddevice as sd
import webrtcvad


SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480 samples
# webrtcvad aggressiveness: 0=permissive, 3=very strict. 2 is a good default
# for indoor desktop use — strict enough to ignore fan noise, lenient enough
# to keep quiet speech.
VAD_AGGRESSIVENESS = 2

# pyautogui per-character interval when typing the transcript. Small but
# non-zero so the URL bar reliably ingests each char even on a busy CPU.
TYPE_INTERVAL_S = 0.005


class VoiceSearch:
    """Mic + VAD + Whisper + typing orchestrator. One instance per app.

    Usage:
        vs = VoiceSearch(model=whisper_model)
        vs.start(on_done=lambda result: ...)   # non-blocking; spawns daemon thread

    ``on_done(result)`` is invoked from the worker thread with one of:
        "ok"         transcript typed + Enter pressed
        "empty"      transcript was empty/whitespace; nothing typed
        "timeout"    no speech detected before initial_silence_frames elapsed
        "mic_error"  mic stream could not be opened
        "error"      unexpected exception during transcription
    """

    def __init__(self, model):
        self.model = model
        self.is_active = False
        self._lock = threading.Lock()

    def start(self, on_done):
        with self._lock:
            if self.is_active:
                return
            self.is_active = True
        threading.Thread(
            target=self._run, args=(on_done,), daemon=True
        ).start()

    def _run(self, on_done):
        try:
            result = self._record_and_transcribe()
        except Exception as exc:
            print(f"[voice_search] unexpected error: {exc!r}")
            result = "error"
        finally:
            with self._lock:
                self.is_active = False
        try:
            on_done(result)
        except Exception as exc:
            print(f"[voice_search] on_done callback raised: {exc!r}")

    def _record_and_transcribe(self):
        detector = SilenceDetector()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        frames_q = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            # indata is (FRAME_SAMPLES, 1) int16. Copy bytes for VAD.
            frames_q.put(bytes(indata))

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=audio_callback,
            )
        except Exception as exc:
            print(f"[voice_search] mic error: {exc!r}")
            return "mic_error"

        captured_frames = []
        with stream:
            while True:
                try:
                    frame_bytes = frames_q.get(timeout=1.0)
                except queue.Empty:
                    # Stream stalled; treat as timeout.
                    return "timeout"
                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
                detector.feed(is_speech)
                if detector.phase is Phase.IN_SPEECH or len(captured_frames) > 0:
                    # Begin accumulating from the first IN_SPEECH transition
                    # onward (including the trailing silence — we don't
                    # bother trimming since Whisper handles leading/trailing
                    # silence fine and the buffer is short).
                    captured_frames.append(frame_bytes)
                if detector.phase is Phase.DONE:
                    break
                if detector.phase is Phase.TIMEOUT:
                    return "timeout"

        # Reconstruct the int16 PCM and convert to float32 normalized to [-1, 1]
        # — the format faster-whisper's .transcribe() accepts via numpy array.
        pcm = np.frombuffer(b"".join(captured_frames), dtype=np.int16)
        audio_f32 = pcm.astype(np.float32) / 32768.0

        segments, _info = self.model.transcribe(
            audio_f32,
            language="en",
            vad_filter=False,  # we already did VAD
        )
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            return "empty"

        pyautogui.write(text, interval=TYPE_INTERVAL_S)
        # Small gap so the URL bar finishes ingesting before Enter commits.
        time.sleep(0.05)
        pyautogui.press("enter")
        return "ok"
```

- [ ] **Step 2: Sanity-import**

Run: `python -c "from handvol.voice_search import VoiceSearch, SilenceDetector, Phase; print('ok')"`
Expected: `ok` (requires `pip install faster-whisper sounddevice webrtcvad` — install before running).

If imports fail because dependencies aren't installed yet, complete Task 5 first, then return here.

- [ ] **Step 3: Run the existing tests to confirm no regression**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (no behavior change to `SilenceDetector`).

- [ ] **Step 4: Commit**

```bash
git add handvol/voice_search.py
git commit -m "Add VoiceSearch orchestrator (mic + VAD + Whisper + typing)"
```

---

## Task 5: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:

```
faster-whisper>=1.0.0
sounddevice>=0.4.6
webrtcvad>=2.0.10
```

- [ ] **Step 2: Install them**

Run: `pip install -r requirements.txt`
Expected: faster-whisper, sounddevice, webrtcvad installed. (faster-whisper will pull ctranslate2 and tokenizers as transitive deps. The first `model.transcribe()` invocation in the app will download the `base.en` model — about 140 MB.)

- [ ] **Step 3: Sanity-import**

Run: `python -c "import faster_whisper, sounddevice, webrtcvad; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add faster-whisper, sounddevice, webrtcvad dependencies"
```

---

## Task 6: Wire dispatch in handvol.pyw

**Files:**
- Modify: `handvol.pyw`

This task does three things: (a) adds `make_mic_image()` for the tray glyph, (b) loads the Whisper model once at startup, (c) dispatches `Event.VOICE_SEARCH` in `capture_loop` and polls a shared `voice_state` dict for completion to restore the lock + tray icon.

- [ ] **Step 1: Add make_mic_image() helper**

In `handvol.pyw`, after `make_volume_image()` (after line 95), add:

```python
def make_mic_image():
    """Render a microphone glyph on a transparent square. Shown in the tray
    while voice search is recording."""
    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Mic capsule (body): rounded rectangle centered upper-half
    cap_w = ICON_SIZE // 3
    cap_h = ICON_SIZE // 2
    cx = ICON_SIZE // 2
    cap_top = ICON_SIZE // 6
    red = (255, 50, 50, 255)
    d.rounded_rectangle(
        [cx - cap_w // 2, cap_top, cx + cap_w // 2, cap_top + cap_h],
        radius=cap_w // 2,
        fill=red,
    )
    # Stand: vertical line + horizontal base
    stand_top = cap_top + cap_h + 4
    stand_bottom = ICON_SIZE - 8
    d.line([(cx, stand_top), (cx, stand_bottom)], fill=red, width=4)
    base_w = cap_w
    d.line(
        [(cx - base_w // 2, stand_bottom), (cx + base_w // 2, stand_bottom)],
        fill=red,
        width=4,
    )
    return img
```

- [ ] **Step 2: Import VoiceSearch and the new event**

Update the imports at the top of `handvol.pyw`. Change line 14 from:

```python
from handvol.state import GestureStateMachine, State, Event, HOLD_SECONDS, NUMBER_9
```

to:

```python
from handvol.state import GestureStateMachine, State, Event, HOLD_SECONDS, NUMBER_9, NUMBER_6
```

Add a new import line after line 14:

```python
from handvol.voice_search import VoiceSearch
```

- [ ] **Step 3: Update the lock-passthrough so NUMBER_6 also gets through**

In `capture_loop` (line 132), change:

```python
effective_gesture = gesture if (not locked or gesture == NUMBER_9) else "None"
```

to:

```python
# When locked, only NUMBER_9 (unlock toggle) and NUMBER_6 (voice search,
# which re-uses the lock) need to reach the state machine. Everything else
# is gated.
effective_gesture = gesture if (not locked or gesture in (NUMBER_9, NUMBER_6)) else "None"
```

Note: this preserves the existing behavior — voice search only fires from an unlocked state, but we still pass NUMBER_6 through so a misfire while locked doesn't accumulate stale counters. The actual dispatch below checks `voice_state["active"]` and ignores re-entries.

- [ ] **Step 4: Add voice_state and voice_search to capture_loop's locals**

In `capture_loop`, after the existing `locked = False` line (around line 118), add:

```python
    voice_state = {"active": False, "was_locked": False, "completed": False}

    # Lazy-load: the model is constructed in main() and passed in (see Step 7).
    # Here we just capture the reference from the module-level holder.
    voice_search = _voice_search_holder.get("instance")

    def on_voice_done(result):
        # Called from the VoiceSearch worker thread. Don't mutate shared
        # state here beyond the dict flag — the capture loop polls and
        # applies the restoration on its own thread.
        if args.debug:
            print(f"  voice search done: {result}")
        voice_state["completed"] = True
```

- [ ] **Step 5: Add the VOICE_SEARCH dispatch branch**

Inside the event dispatch chain in `capture_loop`, after the `Event.PAUSE_CAMERA` branch (around line 207), add:

```python
            elif event is Event.VOICE_SEARCH:
                if voice_search is None:
                    if args.debug:
                        print("  voice search unavailable (model failed to load)")
                elif voice_state["active"]:
                    if args.debug:
                        print("  voice search already active — ignoring")
                else:
                    voice_state["active"] = True
                    voice_state["was_locked"] = locked
                    locked = True
                    if args.debug:
                        print("  voice search start: focus chrome + ctrl+t")
                    taskbar.focus_slot(CHROME_TASKBAR_SLOT, presses=1)
                    shortcuts.open_new_tab()
                    icon.icon = make_mic_image()
                    last_rendered_vol = None  # force icon refresh after restore
                    voice_search.start(on_done=on_voice_done)
```

- [ ] **Step 6: Poll for completion each iteration**

Right after the existing volume update block (after the `icon.icon = make_volume_image(...)` block ending around line 251), add:

```python
            if voice_state["completed"]:
                voice_state["completed"] = False
                voice_state["active"] = False
                locked = voice_state["was_locked"]
                last_rendered_vol = None  # force tray glyph re-render on next tick
                if vol_now is not None:
                    icon.icon = make_volume_image(int(round(vol_now)), locked=locked)
```

- [ ] **Step 7: Load Whisper model once at startup**

Near the top of `handvol.pyw`, after the `WINDOW_TITLE` constant (around line 19), add:

```python
# Holder is filled in by main() after the WhisperModel is loaded. Stays None
# if the import or model load fails — in that case voice search is disabled
# and the rest of the app works normally.
_voice_search_holder = {"instance": None}
```

In `main()`, before `start_worker()` (around line 391), add:

```python
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        _voice_search_holder["instance"] = VoiceSearch(model=whisper_model)
        print("[handvol] voice search ready (faster-whisper base.en, int8 CPU)")
    except Exception as exc:
        print(f"[handvol] voice search disabled: {exc!r}")
```

- [ ] **Step 8: Add VOICE_SEARCH to the debug-print event whitelist**

In `capture_loop`'s event-debug block (around line 290), add `Event.VOICE_SEARCH` to the tuple:

```python
                Event.PAUSE_CAMERA, Event.VOICE_SEARCH,
```

- [ ] **Step 9: Run the existing unit tests**

Run: `python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 10: Manual smoke test**

Run: `python handvol.pyw --show --debug`

In the preview window or via gestures:
1. Form NUMBER_6 (open palm + pointer) with two hands for ~5 frames.
2. Expect: console prints `voice search start: focus chrome + ctrl+t`, Chrome focuses + a new tab opens, the tray icon turns into a red mic, the lock indicator turns red in the overlay.
3. Say something short like "best ramen near me".
4. Stop talking; after ~1 second, expect: text appears in Chrome's URL bar, Enter fires, the search runs, the tray icon returns to the volume number, the lock indicator clears.
5. Repeat with no speech at all — after ~5 seconds, expect timeout: tray + lock restore without typing anything.

- [ ] **Step 11: Commit**

```bash
git add handvol.pyw
git commit -m "Wire VOICE_SEARCH dispatch + mic tray glyph in capture loop"
```

---

## Task 7: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Fill the number_6 row in the gesture table**

In `README.md`, change line 25 from:

```
| Open palm + Pointer | Number 6 |
```

to:

```
| Open palm + Pointer | Number 6 — Voice search (focus Chrome, Ctrl+T, dictate, auto-Enter after 1s silence) |
```

- [ ] **Step 2: Add a "Voice Search" subsection to Extra Features**

In `README.md`, after the "Camera Release" paragraph (around line 52), add:

```markdown
**Voice Search:** Form `number_6` (open palm + pointer) to focus Chrome,
open a new tab, and start dictating. Your speech is transcribed locally
with `faster-whisper` (`base.en`, int8, CPU) — no cloud calls. The tray
icon turns into a red microphone and HandVol auto-locks gestures while
recording. After 1 second of silence, the transcript is typed into the
URL bar and Enter fires automatically. If no speech is detected within
~5 seconds, the trigger times out cleanly with no typing.

First invocation downloads the `base.en` model (~140 MB) into the
HuggingFace cache.
```

- [ ] **Step 3: Add a row to the dependency / setup section if needed**

Verify the existing Setup section already covers `pip install -r requirements.txt`. No further edits needed there — the new entries in `requirements.txt` handle it.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document voice search in README"
```

---

## Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS, including the new `test_voice_search.py` and the `test_scrubber.py` additions.

- [ ] **Step 2: End-to-end smoke test (manual)**

Re-run the manual smoke test from Task 6 Step 10. Confirm:
- Happy path: voice search → text in URL bar → search fires.
- Timeout path: no speech → graceful restore.
- Empty-transcript path: speak gibberish that Whisper transcribes as empty → no Enter, graceful restore.
- Lock preservation: lock manually with NUMBER_9 first, then trigger NUMBER_6 — confirm lock remains on after voice search completes.

- [ ] **Step 3: Push the branch**

```bash
git push origin voice-search
```

---

## Notes for the engineer

- **Whisper first-run latency:** the first call to `model.transcribe()` after process start can take a few seconds while CTranslate2 warms up its kernels. Subsequent calls on short clips are sub-second on a modern CPU. This is fine for our use case.
- **Webrtcvad on Windows:** `pip install webrtcvad` may need build tools. If install fails, `pip install webrtcvad-wheels` is a drop-in replacement with prebuilt wheels.
- **Pyautogui modifier hygiene:** every key-event sequence (`Ctrl+T`, the `pyautogui.write()`, the final `pyautogui.press("enter")`) must release modifiers in a `finally` block on the existing helpers. `pyautogui.write()` and `pyautogui.press()` do this internally; the explicit `keyDown/keyUp` pattern we added in `open_new_tab()` follows `shortcuts.close_window()`.
- **Why polling `voice_state` instead of a queue:** the capture loop already runs at ~30 fps, so a dict-flag check each iteration adds zero overhead and avoids introducing a queue dependency. The flag is only written once (worker → True) and read once (loop → False after acting), so no race even without explicit locking.
