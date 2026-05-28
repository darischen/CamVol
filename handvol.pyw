import argparse
import subprocess
import threading
import time
from collections import deque

from PIL import Image, ImageDraw, ImageFont
from pystray import Icon, Menu, MenuItem

from handvol import audio, media, spotify, taskbar, vscode, shortcuts
from handvol import discord as discord_app
from handvol.capture import GestureSource, MODEL_PATH
from handvol.scrubber import VolumeScrubber
from handvol.state import GestureStateMachine, State, Event, HOLD_SECONDS, NUMBER_9, NUMBER_6
from handvol.voice_search import VoiceSearch


INDEX_TIP = 8  # MediaPipe landmark index for the index fingertip
THUMB_TIP = 4  # MediaPipe landmark index for the thumb tip
WINDOW_TITLE = "HandVol"

# Holder is filled in by main() after the WhisperModel is loaded. Stays None
# if the import or model load fails — in that case voice search is disabled
# and the rest of the app works normally.
_voice_search_holder = {"instance": None}

# Taskbar slot Chrome is pinned to. Number_1 sends Win+<slot> once (focus/launch
# the first window); Number_2 sends it twice with Win held (cycle to the second
# window). Change this if you re-pin Chrome.
CHROME_TASKBAR_SLOT = 1


def scrub_tip(landmarks):
    """Return (x, y) normalized coords at the midpoint of the OK pinch."""
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    return ((thumb.x + index.x) / 2.0, (thumb.y + index.y) / 2.0)


def parse_args():
    p = argparse.ArgumentParser(description="HandVol — gesture-controlled volume")
    p.add_argument("--sensitivity", type=float, default=80.0,
                   help="Volume points per full-frame vertical travel (default 80)")
    p.add_argument("--smoothing", type=float, default=0.3,
                   help="EMA factor on tip Y; lower=lag, higher=jitter (default 0.3)")
    p.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--debug", action="store_true",
                   help="Print frame-by-frame state + values")
    p.add_argument("--no-audio", action="store_true",
                   help="Skip pycaw/media calls — overlay only (useful for tuning)")
    p.add_argument("--show", action="store_true",
                   help="Start with the preview window open. Otherwise tray-only.")
    return p.parse_args()


ICON_SIZE = 64  # Windows downsamples to 16/32; render large for sharp edges.
_FONT_CACHE = {}


def _load_font(size):
    """Load a bold sans font shipped with Windows; fall back to PIL default."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for name in ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def make_volume_image(level, dimmed=False, locked=False):
    """Render the integer volume (0-100) centered on a transparent square.
    Sized so that 3 digits ('100') still fit; 1- and 2-digit values use a
    larger glyph for legibility at 16x16. When dimmed=True, the text is
    drawn in semi-transparent gray to signal a paused state. When locked=True,
    the text is drawn in red to signal a locked state."""
    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    text = str(int(level))
    # Bigger font for 1-2 digits; smaller so '100' doesn't clip.
    px = 56 if len(text) <= 2 else 42
    font = _load_font(px)
    # Pillow's textbbox gives the actual rendered bounds incl. ascender offset.
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (ICON_SIZE - w) // 2 - bbox[0]
    y = (ICON_SIZE - h) // 2 - bbox[1]
    if locked:
        fill = (255, 0, 0, 255)
    elif dimmed:
        fill = (128, 128, 128, 180)
    else:
        fill = (255, 255, 255, 255)
    d.text((x, y), text, fill=fill, font=font)
    return img


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


def capture_loop(args, show_evt, worker_stop, icon, request_pause):
    """Runs on a worker thread. Owns the camera, the model, and (when toggled
    on) the OpenCV window. cv2 + overlay are imported here so we only pay the
    cost when the worker actually starts. worker_stop is signaled both for
    full app quit and for pause — the loop exits the GestureSource context
    cleanly either way, releasing the camera."""
    import cv2
    from handvol.overlay import (
        draw_state, draw_gesture, draw_volume, draw_fps,
        draw_landmarks, draw_scrub_indicator, draw_hold_timer, draw_lock_state,
    )

    scrubber = VolumeScrubber(sensitivity=args.sensitivity, smoothing=args.smoothing)
    machine = GestureStateMachine()

    fps_window = deque(maxlen=30)
    last_t = time.monotonic()
    last_state = None
    window_open = False
    last_rendered_vol = None
    locked = False

    voice_state = {"active": False, "was_locked": False}
    # threading.Event gives an explicit happens-before for the cross-thread
    # handoff. The VoiceSearch daemon sets it; the capture loop checks and
    # clears it each iteration. The cooldown in the state machine already
    # prevents a second VOICE_SEARCH dispatch within the same restoration
    # window, but Event makes the synchronization explicit either way.
    voice_done_evt = threading.Event()

    # Lazy-load: the model is constructed in main() and stored on the
    # module-level holder. Read the reference once when the loop starts.
    voice_search = _voice_search_holder.get("instance")

    def on_voice_done(result):
        # Invoked on the VoiceSearch worker thread.
        if args.debug:
            print(f"  voice search done: {result}")
        voice_done_evt.set()

    with GestureSource(cam_index=args.cam) as source:
        while not worker_stop.is_set():
            frame, latest = source.read()
            if frame is None:
                break

            gesture, score, landmarks, all_landmarks = (
                latest if latest else ("None", 0.0, None, [])
            )
            # When locked, drop every gesture except NUMBER_9 so the state
            # machine sees a stream of "None" and only the unlock toggle
            # gets through — same trick face-recognition uses. NUMBER_6
            # (voice search) is intentionally gated so the lock fully
            # suppresses it.
            effective_gesture = gesture if (not locked or gesture == NUMBER_9) else "None"
            event = machine.step(effective_gesture)

            if event is Event.ENTER_SCRUB and landmarks is not None:
                _, tip_y = scrub_tip(landmarks)
                current_vol = 50.0 if args.no_audio else audio.get_volume()
                scrubber.enter(tip_y, current_vol)

            elif event is Event.UPDATE_SCRUB and landmarks is not None and scrubber.active:
                _, tip_y = scrub_tip(landmarks)
                new_vol = scrubber.update(tip_y)
                if not args.no_audio:
                    audio.set_volume(new_vol)

            elif event is Event.EXIT_SCRUB:
                scrubber.exit()

            elif event is Event.TOGGLE_MUTE:
                if not args.no_audio:
                    audio.toggle_mute()

            elif event is Event.TOGGLE_PLAYPAUSE:
                if not args.no_audio:
                    media.play_pause()

            elif event is Event.TOGGLE_PREVIEW:
                if show_evt.is_set():
                    show_evt.clear()
                else:
                    show_evt.set()

            elif event is Event.FOCUS_SPOTIFY:
                result = spotify.focus_or_launch()
                if args.debug:
                    print(f"  spotify: {result}")

            elif event is Event.EXIT_SPOTIFY:
                result = spotify.exit_spotify()
                if args.debug:
                    print(f"  spotify exit: {result}")

            elif event is Event.FOCUS_DISCORD:
                result = discord_app.focus_or_launch()
                if args.debug:
                    print(f"  discord: {result}")

            elif event is Event.FOCUS_VSCODE:
                result = vscode.focus_or_launch()
                if args.debug:
                    print(f"  vscode: {result}")

            elif event is Event.FOCUS_CHROME_1:
                result = taskbar.focus_slot(CHROME_TASKBAR_SLOT, presses=1)
                if args.debug:
                    print(f"  chrome 1: {result}")

            elif event is Event.FOCUS_CHROME_2:
                result = taskbar.focus_slot(CHROME_TASKBAR_SLOT, presses=2)
                if args.debug:
                    print(f"  chrome 2: {result}")

            elif event is Event.OPEN_TASK_MANAGER:
                result = shortcuts.open_task_manager()
                if args.debug:
                    print(f"  task manager: {result}")

            elif event is Event.CLOSE_WINDOW:
                result = shortcuts.close_window()
                if args.debug:
                    print(f"  close window: {result}")

            elif event is Event.PAUSE_CAMERA:
                if args.debug:
                    print("  pause camera (hang loose)")
                request_pause()

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
                    voice_search.start(on_done=on_voice_done)

            elif event is Event.TOGGLE_LOCK:
                locked = not locked
                if args.debug:
                    print(f"  lock toggled: {'LOCKED' if locked else 'UNLOCKED'}")
                if vol_now is not None:
                    vol_int = int(round(vol_now))
                    icon.icon = make_volume_image(vol_int, locked=locked)

            elif event is Event.NEXT_TRACK:
                if not args.no_audio:
                    media.next_track()

            elif event is Event.PREV_TRACK:
                if not args.no_audio:
                    media.previous_track()

            elif event is Event.RESTART_PC:
                print("[handvol] RESTART_PC fired — running 'shutdown /r /t 0'")
                subprocess.Popen(["shutdown", "/r", "/t", "0"])

            elif event is Event.SHUTDOWN_PC:
                print("[handvol] SHUTDOWN_PC fired — running 'shutdown /s /t 0'")
                subprocess.Popen(["shutdown", "/s", "/t", "0"])

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_window.append(1.0 / dt)
            fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

            try:
                vol_now = None if args.no_audio else audio.get_volume()
            except Exception:
                vol_now = None

            # Push a new tray glyph the moment the displayed integer changes.
            # No throttle — the comparison itself is the rate limit (one update
            # per integer step), and pystray's icon assignment is thread-safe.
            # Skip the volume render while voice search is active so the mic
            # glyph stays put; the completion handler restores it below.
            if vol_now is not None and not voice_state["active"]:
                vol_int = int(round(vol_now))
                if vol_int != last_rendered_vol:
                    icon.icon = make_volume_image(vol_int, locked=locked)
                    last_rendered_vol = vol_int

            if voice_done_evt.is_set():
                voice_done_evt.clear()
                voice_state["active"] = False
                locked = voice_state["was_locked"]
                last_rendered_vol = None  # force tray glyph re-render on next tick
                if vol_now is not None:
                    icon.icon = make_volume_image(int(round(vol_now)), locked=locked)

            want_window = show_evt.is_set()
            if want_window:
                for lm in all_landmarks:
                    draw_landmarks(frame, lm)
                draw_state(frame, machine.state.value)
                draw_gesture(frame, gesture, score)
                hold_progress = machine.get_hold_progress()
                if hold_progress:
                    action, elapsed = hold_progress
                    draw_hold_timer(frame, action, elapsed, HOLD_SECONDS)
                draw_volume(frame, vol_now)
                draw_lock_state(frame, locked)
                if machine.state is State.SCRUB and scrubber.active and landmarks is not None:
                    tip_x, tip_y = scrub_tip(landmarks)
                    draw_scrub_indicator(frame, scrubber.anchor_y, tip_y, tip_x)
                draw_fps(frame, fps)
                cv2.imshow(WINDOW_TITLE, frame)
                window_open = True
                # Esc inside the window or clicking X closes the window and clears the flag
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc
                    show_evt.clear()
                elif cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    # Window was closed by clicking the X button
                    show_evt.clear()
                    window_open = False
            elif window_open:
                cv2.destroyWindow(WINDOW_TITLE)
                # waitKey is needed for destroyWindow to actually process on some builds
                cv2.waitKey(1)
                window_open = False

            if machine.state != last_state or event is not Event.NONE:
                if args.debug or event in (
                    Event.ENTER_SCRUB, Event.EXIT_SCRUB,
                    Event.TOGGLE_MUTE, Event.TOGGLE_PLAYPAUSE, Event.TOGGLE_PREVIEW,
                    Event.FOCUS_SPOTIFY, Event.EXIT_SPOTIFY, Event.FOCUS_DISCORD,
                    Event.FOCUS_VSCODE,
                    Event.FOCUS_CHROME_1, Event.FOCUS_CHROME_2,
                    Event.NEXT_TRACK, Event.PREV_TRACK,
                    Event.RESTART_PC, Event.SHUTDOWN_PC, Event.OPEN_TASK_MANAGER, Event.CLOSE_WINDOW,
                    Event.PAUSE_CAMERA, Event.VOICE_SEARCH,
                ):
                    print(f"[{machine.state.value:14s}] gesture={gesture:14s} "
                          f"event={event.value:18s} fps={fps:5.1f}")
                last_state = machine.state

            if args.debug and event is Event.UPDATE_SCRUB and scrubber.active:
                print(f"  scrub anchor_y={scrubber.anchor_y:.3f} "
                      f"smoothed_y={scrubber.smoothed_y:.3f} "
                      f"vol={vol_now if vol_now is not None else 'n/a'}")

    if window_open:
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing model bundle at {MODEL_PATH}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        )

    show_evt = threading.Event()
    if args.show:
        show_evt.set()

    # Mutable holders so the closures below can rebuild the worker on resume
    # without nonlocal gymnastics.
    paused = {"v": False}
    worker_state = {"stop": None, "thread": None}

    def start_worker():
        worker_state["stop"] = threading.Event()
        worker_state["thread"] = threading.Thread(
            target=capture_loop,
            args=(args, show_evt, worker_state["stop"], icon, lambda: on_pause(icon, None)),
            daemon=True)
        worker_state["thread"].start()

    def stop_worker():
        if worker_state["stop"] is not None:
            worker_state["stop"].set()
        # Skip the join when called from the worker itself (gesture-triggered
        # pause) — joining your own thread raises RuntimeError. The loop will
        # exit cleanly on its next iteration anyway once worker_stop is set.
        if (worker_state["thread"] is not None
                and worker_state["thread"] is not threading.current_thread()):
            worker_state["thread"].join(timeout=2.0)

    def on_toggle(icon, item):
        if show_evt.is_set():
            show_evt.clear()
        else:
            show_evt.set()

    def on_pause(icon, item):
        if paused["v"]:
            paused["v"] = False
            start_worker()
        else:
            paused["v"] = True
            # Close the preview too — camera is going away anyway.
            show_evt.clear()
            stop_worker()
            try:
                vol = 0 if args.no_audio else int(round(audio.get_volume()))
            except Exception:
                vol = 0
            icon.icon = make_volume_image(vol, dimmed=True)
        # Force pystray to re-evaluate the `checked` lambdas. Without this, the
        # tray menu's check state can lag behind paused["v"] when the flip was
        # triggered by the gesture rather than a click.
        icon.update_menu()

    def on_quit(icon, item):
        stop_worker()
        icon.stop()

    menu = Menu(
        MenuItem("Show preview", on_toggle, default=True,
                 checked=lambda item: show_evt.is_set()),
        MenuItem("Pause", on_pause,
                 checked=lambda item: paused["v"]),
        MenuItem("Quit", on_quit),
    )
    # Initial glyph: show the current volume immediately so the icon isn't
    # blank during the worker's first-frame warmup (which can take ~1s while
    # MediaPipe loads).
    try:
        initial_vol = 0 if args.no_audio else int(round(audio.get_volume()))
    except Exception:
        initial_vol = 0
    icon = Icon("handvol", make_volume_image(initial_vol), "HandVol", menu)

    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
        _voice_search_holder["instance"] = VoiceSearch(model=whisper_model)
        print("[handvol] voice search ready (faster-whisper small.en, int8 CPU)")
    except Exception as exc:
        print(f"[handvol] voice search disabled: {exc!r}")

    start_worker()

    # pystray blocks the main thread running the Win32 message pump until
    # icon.stop() is called.
    icon.run()

    stop_worker()


if __name__ == "__main__":
    main()
