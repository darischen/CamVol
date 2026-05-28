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
