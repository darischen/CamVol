import argparse
import threading
import time
from collections import deque

from PIL import Image, ImageDraw, ImageFont
from pystray import Icon, Menu, MenuItem

from handvol import audio, media, spotify
from handvol.capture import GestureSource, MODEL_PATH
from handvol.scrubber import VolumeScrubber
from handvol.state import GestureStateMachine, State, Event

import subprocess
import sys

from handvol.face_profile import FaceProfile, DEFAULT_PROFILE_PATH
from handvol.face_detect import FACE_MODEL_FILENAME


INDEX_TIP = 8  # MediaPipe landmark index for the index fingertip
THUMB_TIP = 4  # MediaPipe landmark index for the thumb tip
WINDOW_TITLE = "HandVol"


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


def make_volume_image(level, dimmed=False):
    """Render the integer volume (0-100) centered on a transparent square.
    Sized so that 3 digits ('100') still fit; 1- and 2-digit values use a
    larger glyph for legibility at 16x16. When dimmed=True, the text is
    drawn in semi-transparent gray to signal a paused state."""
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
    fill = (128, 128, 128, 180) if dimmed else (255, 255, 255, 255)
    d.text((x, y), text, fill=fill, font=font)
    return img


def capture_loop(args, show_evt, worker_stop, icon, profile_state):
    """Runs on a worker thread. Owns the camera, the model, and (when toggled
    on) the OpenCV window. cv2 + overlay are imported here so we only pay the
    cost when the worker actually starts. worker_stop is signaled both for
    full app quit and for pause — the loop exits the GestureSource context
    cleanly either way, releasing the camera."""
    import cv2
    from handvol.overlay import (
        draw_state, draw_gesture, draw_volume, draw_fps,
        draw_landmarks, draw_scrub_indicator, draw_lock_state,
    )

    scrubber = VolumeScrubber(sensitivity=args.sensitivity, smoothing=args.smoothing)
    machine = GestureStateMachine()

    fps_window = deque(maxlen=30)
    last_t = time.monotonic()
    last_state = None
    window_open = False
    last_rendered_vol = None
    NO_FACE_GRACE_FRAMES = 15
    no_face_streak = NO_FACE_GRACE_FRAMES  # start locked until proven otherwise
    last_recognized = False

    with GestureSource(cam_index=args.cam) as source:
        while not worker_stop.is_set():
            frame, latest = source.read()
            if frame is None:
                break

            if latest is None:
                gesture, score, landmarks, face_embs = ("None", 0.0, None, [])
            else:
                gesture, score, landmarks, face_embs = latest

            profile = profile_state["profile"]
            if profile is None or profile.capture_count == 0:
                recognized = False
            elif not face_embs:
                no_face_streak += 1
                recognized = last_recognized if no_face_streak < NO_FACE_GRACE_FRAMES else False
            else:
                no_face_streak = 0
                # Spec: if ANY face in frame matches the profile, unlock.
                recognized = any(profile.matches(e)[0] for e in face_embs)
            last_recognized = recognized

            # Drop gesture if the user is not recognized; the state machine
            # then sees a stream of "None" and gracefully exits any active
            # SCRUB / etc. without us having to special-case it.
            effective_gesture = gesture if recognized else "None"
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

            elif event is Event.FOCUS_SPOTIFY:
                result = spotify.focus_or_launch()
                if args.debug:
                    print(f"  spotify: {result}")

            elif event is Event.EXIT_SPOTIFY:
                result = spotify.exit_spotify()
                if args.debug:
                    print(f"  spotify exit: {result}")

            elif event is Event.NEXT_TRACK:
                if not args.no_audio:
                    media.next_track()

            elif event is Event.PREV_TRACK:
                if not args.no_audio:
                    media.previous_track()

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
            if vol_now is not None:
                vol_int = int(round(vol_now))
                if vol_int != last_rendered_vol:
                    icon.icon = make_volume_image(vol_int)
                    last_rendered_vol = vol_int

            want_window = show_evt.is_set()
            if want_window:
                if landmarks is not None:
                    draw_landmarks(frame, landmarks)
                draw_state(frame, machine.state.value)
                draw_gesture(frame, gesture, score)
                draw_volume(frame, vol_now)
                draw_lock_state(
                    frame, recognized,
                    has_profile=(profile_state["profile"] is not None
                                 and profile_state["profile"].capture_count > 0),
                )
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
                    Event.TOGGLE_MUTE, Event.TOGGLE_PLAYPAUSE,
                    Event.FOCUS_SPOTIFY, Event.EXIT_SPOTIFY,
                    Event.NEXT_TRACK, Event.PREV_TRACK,
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

    face_model_path = MODEL_PATH.parent / FACE_MODEL_FILENAME
    if not face_model_path.exists():
        raise SystemExit(
            f"Missing face landmarker model at {face_model_path}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    show_evt = threading.Event()
    if args.show:
        show_evt.set()

    # Mutable holder so on_calibrate can swap in a freshly loaded profile.
    profile_state = {"profile": FaceProfile.load(DEFAULT_PROFILE_PATH)}

    # Mutable holders so the closures below can rebuild the worker on resume
    # without nonlocal gymnastics.
    paused = {"v": False}
    worker_state = {"stop": None, "thread": None}

    def start_worker():
        worker_state["stop"] = threading.Event()
        worker_state["thread"] = threading.Thread(
            target=capture_loop,
            args=(args, show_evt, worker_state["stop"], icon, profile_state),
            daemon=True)
        worker_state["thread"].start()

    def stop_worker():
        if worker_state["stop"] is not None:
            worker_state["stop"].set()
        if worker_state["thread"] is not None:
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

    def on_calibrate(icon, item):
        # Stop the capture worker so the camera is free, run calibration
        # in a subprocess (blocks the tray callback thread, which is fine
        # for pystray), then reload the profile and restart the worker.
        stop_worker()
        time.sleep(0.3)  # let DirectShow release the camera handle
        try:
            python = sys.executable
            # If we were launched by pythonw, switch to python for the
            # calibration so OpenCV's imshow window is interactable.
            if python.lower().endswith("pythonw.exe"):
                python = python[: -len("pythonw.exe")] + "python.exe"
            subprocess.run(
                [python, "-m", "handvol.calibration", "--force"],
                check=False,
            )
        finally:
            profile_state["profile"] = FaceProfile.load(DEFAULT_PROFILE_PATH)
            if not paused["v"]:
                start_worker()

    def on_quit(icon, item):
        stop_worker()
        icon.stop()

    menu = Menu(
        MenuItem("Show preview", on_toggle, default=True,
                 checked=lambda item: show_evt.is_set()),
        MenuItem("Pause", on_pause,
                 checked=lambda item: paused["v"]),
        MenuItem("Calibrate face...", on_calibrate),
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

    start_worker()

    # pystray blocks the main thread running the Win32 message pump until
    # icon.stop() is called.
    icon.run()

    stop_worker()


if __name__ == "__main__":
    main()
