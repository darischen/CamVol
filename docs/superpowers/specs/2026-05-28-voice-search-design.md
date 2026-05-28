# Voice Search via number_6 — Design

## Goal

Trigger a fully-offline voice search in Chrome with the `number_6` gesture
(open palm + pointer). The mic listens until you stop talking; after 1
second of silence the transcript is typed into Chrome's URL bar and Enter
is pressed, running the search.

## Trigger and flow (main capture loop)

`number_6` is currently unmapped in `handvol.pyw`. When it fires:

1. Focus or launch Chrome by calling the same Win+1 path used for
   `number_1` (`taskbar.focus_slot(1)`).
2. Set the existing gesture lock (`locked = True`) so subsequent gestures
   are dropped just as if `number_9` had been pressed. Remember whether
   the lock was *already* on before we touched it.
3. Send `Ctrl+T` via pyautogui using the same modifier-warmup +
   try/finally pattern as `shortcuts.py` / `taskbar.py` (see "Key delay
   discipline" below).
4. Swap the tray glyph to a microphone icon (new `make_mic_image()`
   helper next to `make_volume_image()` in `handvol.pyw`).
5. Spawn the voice-search background thread and return. The main capture
   loop keeps running — gestures are simply gated by `locked`.

Re-entry: if `number_6` fires while a voice search is already active, it
is ignored (guarded by an `is_active` flag on the `VoiceSearch`
instance).

## Voice-search thread

New module: `handvol/voice_search.py`. Exposes a `VoiceSearch` class with
roughly this surface:

```python
VoiceSearch(model, on_done)        # model: faster-whisper WhisperModel
voice_search.start()               # no-op if already active
voice_search.is_active             # bool
```

`on_done(result)` is invoked from the worker thread when the search
completes (success, timeout, or empty transcript). The main loop uses it
to clear the lock and restore the tray glyph.

Internal sequence:

1. Open a `sounddevice` input stream: 16 kHz, mono, int16, 30 ms frames
   (the frame size `webrtcvad` requires).
2. **VAD state machine:**
   - `WAITING_FOR_SPEECH`: collect frames; if the cumulative silence
     reaches ~5 s without speech ever being detected, bail out
     (`on_done("timeout")`).
   - `IN_SPEECH`: once VAD has flagged ~10 consecutive voiced frames
     (~300 ms of speech, to debounce stray noise), transition. Keep
     appending frames. Track a rolling silence counter — when it hits
     1.0 s of continuous silence, stop the stream.
3. Transcribe the accumulated buffer with `faster-whisper`
   (`base.en`, `compute_type="int8"`, CPU). The model is loaded **once
   at app startup** in `handvol.pyw` so the first invocation has no warm
   up cost.
4. If the transcript is empty/whitespace, call `on_done("empty")` and
   return without typing anything (no Enter on nothing).
5. Otherwise `pyautogui.write(text, interval=0.005)` to type the
   transcript, then `pyautogui.press("enter")`, then
   `on_done("ok")`.

The thread is a daemon thread so it dies with the process.

## Key delay discipline

To match the existing pattern and avoid stuck modifiers, the `Ctrl+T`
send lives in a small helper (either in `handvol/shortcuts.py` or a new
`open_new_tab()` in the same file):

```python
pyautogui.PAUSE = 0
MODIFIER_WARMUP = 0.05
INTER_KEY_DELAY = 0.05

def open_new_tab():
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

`pyautogui.write()` for the transcript will use a small per-character
`interval` (~5 ms) so Chrome's URL bar reliably ingests each character
even on a busy CPU. `pyautogui.press("enter")` for the final commit.

## Files touched

| File | Change |
|---|---|
| `handvol/voice_search.py` | **New.** `VoiceSearch` class: mic stream, VAD loop, Whisper transcription, typing. |
| `handvol/shortcuts.py` | Add `open_new_tab()` (Ctrl+T) using the existing modifier-warmup pattern. |
| `handvol.pyw` | Load Whisper model at startup; dispatch `number_6` → focus Chrome, set lock, fire Ctrl+T, swap tray glyph, start `VoiceSearch`; add `make_mic_image()`; wire `on_done` to restore lock + glyph. |
| `requirements.txt` | Add `faster-whisper`, `sounddevice`, `webrtcvad`. |
| `README.md` | Fill the `number_6` row; add a short "Voice search" section under Extra Features. |

## Edge cases

- **Whisper empty transcript:** skip typing and Enter. Just clean up.
- **Initial-silence timeout (~5 s):** clean up without typing. Prevents
  a stray trigger from leaving the mic open forever.
- **Lock was already on:** if the user manually locked HandVol with
  `number_9` before triggering `number_6`, leave the lock on at the
  end. We only auto-unlock if we were the ones who set it.
- **Re-entry:** ignore `number_6` while `is_active` is True.
- **Whisper model load failure at startup:** log the error and continue
  without the feature; `number_6` becomes a no-op. The rest of HandVol
  works.
- **Mic device unavailable:** caught in `VoiceSearch.start()`; call
  `on_done("mic_error")` immediately so the lock and tray glyph are
  restored.

## Threading model

The MediaPipe capture loop is sync and tight; we do not want to block it
on Whisper inference (~0.3–1 s for a short query on CPU). The
`VoiceSearch` thread is a daemon, communicates with the main loop only
via the `on_done` callback, and never touches `locked` directly. The
main loop's callback is responsible for:

- Resetting `locked` to its pre-trigger value.
- Restoring `icon.icon` to the current volume glyph.
- Clearing any `is_voice_search_active` flag the loop tracks.

This keeps state ownership clean: the worker owns audio + Whisper +
typing; the main loop owns gesture state and the tray icon.

## Testing

- **Unit:** `tests/test_voice_search_vad.py` — feed synthetic frames
  (alternating speech/silence) into the VAD state-machine logic and
  assert correct transitions (waiting → speaking → done after 1 s
  silence, timeout after 5 s with no speech).
- **Unit:** state-machine wrapper exposed as a pure function for ease of
  testing, separate from the sounddevice / Whisper integration.
- **Manual smoke test:** trigger `number_6`, say "best ramen near me",
  confirm Chrome opens a new tab with that text and Enter fires after ~1
  s of silence; confirm tray glyph cycles mic → volume; confirm lock
  reverts to its prior state.

## Dependencies (one-time install)

```
faster-whisper
sounddevice
webrtcvad
```

`faster-whisper` will download the `base.en` model (~140 MB) on first
use. The download path is `~/.cache/huggingface/...` by default; no
config needed.

## Out of scope

- Streaming partial transcripts to the URL bar (we chose record-then-
  type for simplicity and snappier feel).
- Wake-word activation. The gesture is the trigger.
- Multilingual support. `base.en` is English-only by design.
- A visible recording-time indicator. The tray mic glyph is enough; the
  preview window doesn't need extra UI for this.
