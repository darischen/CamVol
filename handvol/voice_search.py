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
import difflib
import queue
import threading
import time
from enum import Enum

import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
import webrtcvad


MODIFIER_WARMUP = 0.05
INTER_KEY_DELAY = 0.05

# Spoken-command prefixes that mean "navigate", not "search". These are short,
# high-frequency English words Whisper transcribes reliably, so they make a
# dependable intent signal: text starting with one is treated as a destination.
GO_PREFIXES = ("go to", "goto", "open", "navigate to")

# Friendly name (and synonyms) -> URL. Whisper only has to land *near* a key,
# not spell the full domain — fuzzy matching against this closed set snaps a
# garbled "gemoney" back to "gemini". Add common sites here.
ALIASES = {
    "gemini":    "https://gemini.google.com",
    "youtube":   "https://youtube.com",
    "yt":        "https://youtube.com",
    "instagram": "https://instagram.com",
    "insta":     "https://instagram.com",
    "ig":        "https://instagram.com",
    "github":    "https://github.com",
    "gmail":     "https://mail.google.com",
    "drive":     "https://drive.google.com",
    "chatgpt":   "https://chat.openai.com",
    "claude":    "https://claude.ai",
    "reddit":    "https://reddit.com",
    "twitter":   "https://twitter.com",
    "x":         "https://x.com",
    "netflix":   "https://netflix.com",
    "amazon":    "https://amazon.com",
}

# difflib similarity cutoff for the fuzzy alias match: 0 matches anything,
# 1 requires an exact key. 0.6 tolerates Whisper's near-misses while still
# rejecting unrelated words.
FUZZY_CUTOFF = 0.6


def resolve_destination(text):
    """Map a spoken "go to X" command to what should be typed into the omnibox.

    Returns:
        - the matched site URL when X resolves (exact or fuzzy) to a known site;
        - the bare target word when a command prefix is present but X is unknown
          (so the omnibox searches just "spotify", not "go to spotify");
        - ``None`` when there's no command prefix, signalling the caller to use
          the original transcript verbatim.
    """
    lowered = text.lower().strip()
    for prefix in GO_PREFIXES:
        if lowered.startswith(prefix + " "):
            target = lowered[len(prefix):].strip()
            # Drop a spoken or written ".com"/"dot com" tail so "go to youtube
            # dot com" still keys on "youtube".
            target = target.replace("dot com", "").replace(".com", "").strip()
            if not target:
                return None
            if target in ALIASES:
                return ALIASES[target]
            match = difflib.get_close_matches(
                target, ALIASES.keys(), n=1, cutoff=FUZZY_CUTOFF)
            if match:
                return ALIASES[match[0]]
            return target  # known prefix, unknown site: search the bare target
    return None


class Phase(str, Enum):
    WAITING_FOR_SPEECH = "waiting_for_speech"
    IN_SPEECH = "in_speech"
    DONE = "done"
    TIMEOUT = "timeout"


# Defaults assume 30 ms frames (webrtcvad's native frame size at 16 kHz).
# 10 frames * 30 ms = 300 ms speech debounce
# 33 frames * 30 ms approx 1.0 s end-of-utterance silence
# 166 frames * 30 ms approx 5.0 s no-speech timeout
DEFAULT_SPEECH_START_FRAMES = 10
DEFAULT_SILENCE_END_FRAMES = 33
DEFAULT_INITIAL_SILENCE_FRAMES = 166


class SilenceDetector:
    """Pure-logic VAD phase tracker. Feed one is_speech bool per audio frame.

    Phase transitions:
        WAITING_FOR_SPEECH -> IN_SPEECH  after speech_start_frames consecutive speech
        WAITING_FOR_SPEECH -> TIMEOUT    after initial_silence_frames with no speech
        IN_SPEECH          -> DONE       after silence_end_frames consecutive silence

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


SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480 samples
# webrtcvad aggressiveness: 0=permissive, 3=very strict. 2 is a good default
# for indoor desktop use - strict enough to ignore fan noise, lenient enough
# to keep quiet speech.
VAD_AGGRESSIVENESS = 2

# Paste the transcript via clipboard rather than pyautogui.write(): write()
# silently drops non-ASCII chars and uses an implicit Shift modifier for
# punctuation with no try/finally guard — both classes of bug the codebase
# explicitly avoids elsewhere (see handvol/taskbar.py).
pyautogui.PAUSE = 0


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

    def start(self, on_done, resolve=True, press_enter=True):
        """Begin a recording session on a daemon thread.

        ``resolve``     apply the "go to X" alias table (search mode); set False
                        for raw dictation that types the transcript verbatim.
        ``press_enter`` hit Enter after pasting (submit); set False to leave the
                        text sitting in the field for the user to review.
        """
        with self._lock:
            if self.is_active:
                return
            self.is_active = True
        threading.Thread(
            target=self._run, args=(on_done, resolve, press_enter), daemon=True
        ).start()

    def _run(self, on_done, resolve, press_enter):
        try:
            result = self._record_and_transcribe(resolve, press_enter)
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

    def _record_and_transcribe(self, resolve=True, press_enter=True):
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
                    # onward (including the trailing silence - we don't
                    # bother trimming since Whisper handles leading/trailing
                    # silence fine and the buffer is short).
                    captured_frames.append(frame_bytes)
                if detector.phase is Phase.DONE:
                    break
                if detector.phase is Phase.TIMEOUT:
                    return "timeout"

        # Reconstruct the int16 PCM and convert to float32 normalized to [-1, 1]
        # - the format faster-whisper's .transcribe() accepts via numpy array.
        pcm = np.frombuffer(b"".join(captured_frames), dtype=np.int16)
        audio_f32 = pcm.astype(np.float32) / 32768.0

        segments, _info = self.model.transcribe(
            audio_f32,
            language="en",
            vad_filter=False,  # we already did VAD
        )
        text = " ".join(seg.text for seg in segments).strip()
        # Whisper appends sentence-final punctuation that breaks URL queries —
        # "instagram.com." parses as a different domain than "instagram.com".
        # Strip trailing punctuation; Chrome's omnibox handles the rest.
        text = text.rstrip(".,!?;:")
        if not text:
            return "empty"

        # Search mode resolves "go to X" to a URL (or the bare site name on a
        # fuzzy miss) and otherwise pastes verbatim. Dictation mode (resolve
        # False) always types the raw transcript. Either way it's one string
        # down the same paste path.
        to_type = (resolve_destination(text) or text) if resolve else text

        pyperclip.copy(to_type)
        try:
            pyautogui.keyDown("ctrl")
            try:
                time.sleep(MODIFIER_WARMUP)
                pyautogui.press("v")
            finally:
                time.sleep(INTER_KEY_DELAY)
                pyautogui.keyUp("ctrl")
        except Exception as exc:
            print(f"[voice_search] paste failed: {exc!r}")
            return "error"
        if press_enter:
            # Small gap so the field finishes ingesting before Enter commits.
            time.sleep(0.05)
            pyautogui.press("enter")
        return "ok"
