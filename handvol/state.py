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


SCRUB_ENTER_FRAMES = 5
SCRUB_EXIT_FRAMES = 3
TOGGLE_FRAMES = 5
COOLDOWN_FRAMES = 10

POINTING = "Pointing_Up"
FIST = "Closed_Fist"
PALM = "Open_Palm"


class GestureStateMachine:
    """Drives the IDLE / SCRUB / IDLE_COOLDOWN debouncer.

    step(gesture_name) returns an Event the caller should react to.
    The machine never touches audio or media directly.
    """

    def __init__(self):
        self.state = State.IDLE
        self._point_count = 0
        self._non_point_count = 0
        self._fist_count = 0
        self._palm_count = 0
        self._neutral_count = 0
        self._cooldown_left = 0

    def _reset_counters(self):
        self._point_count = 0
        self._non_point_count = 0
        self._fist_count = 0
        self._palm_count = 0
        self._neutral_count = 0

    def _bump(self, gesture):
        self._point_count = self._point_count + 1 if gesture == POINTING else 0
        self._fist_count = self._fist_count + 1 if gesture == FIST else 0
        self._palm_count = self._palm_count + 1 if gesture == PALM else 0
        if gesture in (POINTING, FIST, PALM):
            self._neutral_count = 0
        else:
            self._neutral_count += 1
        if gesture == POINTING:
            self._non_point_count = 0
        else:
            self._non_point_count += 1

    def step(self, gesture):
        gesture = gesture or "None"
        self._bump(gesture)

        if self.state is State.IDLE:
            if self._point_count >= SCRUB_ENTER_FRAMES:
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
            return Event.NONE

        if self.state is State.SCRUB:
            if self._non_point_count >= SCRUB_EXIT_FRAMES:
                self.state = State.IDLE
                self._reset_counters()
                return Event.EXIT_SCRUB
            if gesture == POINTING:
                return Event.UPDATE_SCRUB
            return Event.NONE

        # IDLE_COOLDOWN
        if self._neutral_count >= 1:
            self._cooldown_left -= 1
        if self._cooldown_left <= 0:
            self.state = State.IDLE
            self._reset_counters()
        return Event.NONE
