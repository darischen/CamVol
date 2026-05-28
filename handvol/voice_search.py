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
import queue
import threading
import time
from enum import Enum

import numpy as np
import pyautogui
import sounddevice as sd
import webrtcvad


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
        if not text:
            return "empty"

        pyautogui.write(text, interval=TYPE_INTERVAL_S)
        # Small gap so the URL bar finishes ingesting before Enter commits.
        time.sleep(0.05)
        pyautogui.press("enter")
        return "ok"
