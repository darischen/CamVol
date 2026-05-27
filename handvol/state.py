import time
from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    SCRUB = "SCRUB"
    IDLE_COOLDOWN = "IDLE_COOLDOWN"


class Event(str, Enum):
    NONE = "none"
    ENTER_SCRUB = "enter_scrub"
    UPDATE_SCRUB = "update_scrub"
    EXIT_SCRUB = "exit_scrub"
    TOGGLE_MUTE = "toggle_mute"
    TOGGLE_PLAYPAUSE = "toggle_playpause"
    TOGGLE_PREVIEW = "toggle_preview"
    FOCUS_SPOTIFY = "focus_spotify"
    EXIT_SPOTIFY = "exit_spotify"
    NEXT_TRACK = "next_track"
    PREV_TRACK = "prev_track"
    RESTART_PC = "restart_pc"
    SHUTDOWN_PC = "shutdown_pc"


SCRUB_ENTER_FRAMES = 5
SCRUB_EXIT_FRAMES = 3
TOGGLE_FRAMES = 5
COOLDOWN_FRAMES = 10
# Wall-clock hold for destructive system gestures. Frame-count debounces drift
# with fps; restart/shutdown need a real 5-second deliberation regardless.
HOLD_SECONDS = 5.0

FIST = "Closed_Fist"
PALM = "Open_Palm"
VICTORY = "Victory"
ILOVEYOU = "ILoveYou"
OK_SIGN = "OK_sign"
POINTING_UP = "Pointing_Up"
MIDDLE_FINGER = "middle_finger"
DOUBLE_MIDDLE_FINGER = "double_middle_finger"
LEFT_HAND_THUMB_LEFT = "left_hand_thumb_left"
LEFT_HAND_THUMB_RIGHT = "left_hand_thumb_right"
RIGHT_HAND_THUMB_LEFT = "right_hand_thumb_left"
RIGHT_HAND_THUMB_RIGHT = "right_hand_thumb_right"

SKIP_GESTURES = (LEFT_HAND_THUMB_RIGHT, RIGHT_HAND_THUMB_RIGHT)
PREV_GESTURES = (LEFT_HAND_THUMB_LEFT, RIGHT_HAND_THUMB_LEFT)


class GestureStateMachine:
    """Drives the IDLE / SCRUB / IDLE_COOLDOWN debouncer.

    step(gesture_name) returns an Event the caller should react to.
    The machine never touches audio or media directly.
    """

    def __init__(self):
        self.state = State.IDLE
        self._ok_count = 0
        self._non_ok_count = 0
        self._fist_count = 0
        self._palm_count = 0
        self._victory_count = 0
        self._iloveyou_count = 0
        self._pointer_count = 0
        self._skip_count = 0
        self._prev_count = 0
        self._neutral_count = 0
        self._cooldown_left = 0
        # Wall-clock start times for hold-gestures; None when not currently held.
        self._middle_start_t = None
        self._double_middle_start_t = None

    def _reset_counters(self):
        self._ok_count = 0
        self._non_ok_count = 0
        self._fist_count = 0
        self._palm_count = 0
        self._victory_count = 0
        self._iloveyou_count = 0
        self._pointer_count = 0
        self._skip_count = 0
        self._prev_count = 0
        self._neutral_count = 0
        self._middle_start_t = None
        self._double_middle_start_t = None

    def _bump(self, gesture):
        is_skip = gesture in SKIP_GESTURES
        is_prev = gesture in PREV_GESTURES
        self._ok_count = self._ok_count + 1 if gesture == OK_SIGN else 0
        self._fist_count = self._fist_count + 1 if gesture == FIST else 0
        self._palm_count = self._palm_count + 1 if gesture == PALM else 0
        self._victory_count = self._victory_count + 1 if gesture == VICTORY else 0
        self._iloveyou_count = self._iloveyou_count + 1 if gesture == ILOVEYOU else 0
        self._pointer_count = self._pointer_count + 1 if gesture == POINTING_UP else 0
        self._skip_count = self._skip_count + 1 if is_skip else 0
        self._prev_count = self._prev_count + 1 if is_prev else 0
        # Hold timers: start on first sighting, clear the moment the gesture drops.
        # Mutually exclusive — flipping between single and double restarts the clock.
        now = time.monotonic()
        if gesture == MIDDLE_FINGER:
            if self._middle_start_t is None:
                self._middle_start_t = now
            self._double_middle_start_t = None
        elif gesture == DOUBLE_MIDDLE_FINGER:
            if self._double_middle_start_t is None:
                self._double_middle_start_t = now
            self._middle_start_t = None
        else:
            self._middle_start_t = None
            self._double_middle_start_t = None
        if is_skip or is_prev or gesture in (
            OK_SIGN, FIST, PALM, VICTORY, ILOVEYOU, POINTING_UP,
            MIDDLE_FINGER, DOUBLE_MIDDLE_FINGER,
        ):
            self._neutral_count = 0
        else:
            self._neutral_count += 1
        if gesture == OK_SIGN:
            self._non_ok_count = 0
        else:
            self._non_ok_count += 1

    def get_hold_progress(self):
        """Return (action_label, elapsed_seconds) if a hold is in progress, else None.

        Used by the overlay to display a timer and action label during holds.
        """
        now = time.monotonic()
        if self._middle_start_t is not None:
            return ("RESTART", now - self._middle_start_t)
        if self._double_middle_start_t is not None:
            return ("SHUTDOWN", now - self._double_middle_start_t)
        return None

    def step(self, gesture):
        gesture = gesture or "None"
        self._bump(gesture)

        if self.state is State.IDLE:
            now = time.monotonic()
            if (self._middle_start_t is not None
                    and now - self._middle_start_t >= HOLD_SECONDS):
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.RESTART_PC
            if (self._double_middle_start_t is not None
                    and now - self._double_middle_start_t >= HOLD_SECONDS):
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.SHUTDOWN_PC
            if self._ok_count >= SCRUB_ENTER_FRAMES:
                self.state = State.SCRUB
                self._reset_counters()
                return Event.ENTER_SCRUB
            if self._fist_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.TOGGLE_MUTE
            if self._palm_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.TOGGLE_PLAYPAUSE
            if self._pointer_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.TOGGLE_PREVIEW
            if self._victory_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.FOCUS_SPOTIFY
            if self._iloveyou_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.EXIT_SPOTIFY
            if self._skip_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.NEXT_TRACK
            if self._prev_count >= TOGGLE_FRAMES:
                self.state = State.IDLE_COOLDOWN
                self._cooldown_left = COOLDOWN_FRAMES
                self._reset_counters()
                return Event.PREV_TRACK
            return Event.NONE

        if self.state is State.SCRUB:
            if self._non_ok_count >= SCRUB_EXIT_FRAMES:
                self.state = State.IDLE
                self._reset_counters()
                return Event.EXIT_SCRUB
            if gesture == OK_SIGN:
                return Event.UPDATE_SCRUB
            return Event.NONE

        # IDLE_COOLDOWN
        if self._neutral_count >= 1:
            self._cooldown_left -= 1
        if self._cooldown_left <= 0:
            self.state = State.IDLE
            self._reset_counters()
        return Event.NONE
