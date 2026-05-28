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
    # only 2 consecutive speech frames so far -> still waiting
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
    # Total of 4 consecutive silence frames -> DONE
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
