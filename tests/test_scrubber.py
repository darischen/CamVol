from handvol.scrubber import VolumeScrubber
from handvol.state import GestureStateMachine, State, Event, OK_SIGN, FIST, PALM


def test_enter_anchors_and_first_update_is_noop():
    s = VolumeScrubber(sensitivity=80, smoothing=0.3)
    s.enter(tip_y=0.5, current_vol=40)
    assert s.update(0.5) == 40


def test_finger_up_raises_volume():
    s = VolumeScrubber(sensitivity=80, smoothing=1.0)  # no smoothing for determinism
    s.enter(tip_y=0.8, current_vol=20)
    # tip_y smaller => finger higher in frame => volume up
    assert s.update(0.3) == 20 + 80 * (0.8 - 0.3)


def test_finger_down_lowers_volume():
    s = VolumeScrubber(sensitivity=80, smoothing=1.0)
    s.enter(tip_y=0.2, current_vol=60)
    assert s.update(0.7) == 60 + 80 * (0.2 - 0.7)


def test_clamp_to_0_100():
    s = VolumeScrubber(sensitivity=80, smoothing=1.0)
    s.enter(tip_y=0.5, current_vol=90)
    assert s.update(0.0) == 100  # would be 130
    s.enter(tip_y=0.5, current_vol=10)
    assert s.update(1.0) == 0  # would be -30


def test_state_machine_enter_scrub_after_5_ok_sign():
    sm = GestureStateMachine()
    events = [sm.step(OK_SIGN) for _ in range(5)]
    assert sm.state is State.SCRUB
    assert events[-1] is Event.ENTER_SCRUB
    assert all(e is Event.NONE for e in events[:-1])


def test_state_machine_exits_scrub_after_3_non_ok_sign():
    sm = GestureStateMachine()
    for _ in range(5):
        sm.step(OK_SIGN)
    assert sm.state is State.SCRUB
    sm.step("None")
    sm.step("None")
    e = sm.step("None")
    assert e is Event.EXIT_SCRUB
    assert sm.state is State.IDLE


def test_fist_toggles_mute_once_then_cooldown():
    sm = GestureStateMachine()
    events = [sm.step(FIST) for _ in range(5)]
    assert events[-1] is Event.TOGGLE_MUTE
    assert sm.state is State.IDLE_COOLDOWN
    # Holding fist during cooldown does not retrigger
    for _ in range(5):
        assert sm.step(FIST) is Event.NONE


def test_palm_toggles_playpause_once():
    sm = GestureStateMachine()
    events = [sm.step(PALM) for _ in range(5)]
    assert events[-1] is Event.TOGGLE_PLAYPAUSE
    assert sm.state is State.IDLE_COOLDOWN


def test_state_machine_number_6_fires_voice_search():
    from handvol.state import NUMBER_6, TOGGLE_FRAMES
    sm = GestureStateMachine()
    events = [sm.step(NUMBER_6) for _ in range(TOGGLE_FRAMES)]
    assert events[-1] is Event.VOICE_SEARCH
    assert all(e is Event.NONE for e in events[:-1])


def test_state_machine_number_6_resets_after_cooldown():
    from handvol.state import NUMBER_6, TOGGLE_FRAMES
    sm = GestureStateMachine()
    for _ in range(TOGGLE_FRAMES):
        sm.step(NUMBER_6)
    # After firing once, machine enters IDLE_COOLDOWN; should not refire immediately.
    next_events = [sm.step(NUMBER_6) for _ in range(TOGGLE_FRAMES)]
    assert Event.VOICE_SEARCH not in next_events
