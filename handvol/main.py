import argparse
import time
from collections import deque

import cv2

from handvol import audio, media
from handvol.capture import GestureSource, MODEL_PATH
from handvol.overlay import (
    draw_state, draw_gesture, draw_volume, draw_fps,
    draw_landmarks, draw_scrub_indicator,
)
from handvol.scrubber import VolumeScrubber
from handvol.state import GestureStateMachine, State, Event


INDEX_TIP = 8  # MediaPipe landmark index for the index fingertip


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
    p.add_argument("--headless", action="store_true",
                   help="Run without the OpenCV window. Quit with Ctrl+Shift+Q.")
    return p.parse_args()


def main():
    args = parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing model bundle at {MODEL_PATH}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        )

    scrubber = VolumeScrubber(sensitivity=args.sensitivity, smoothing=args.smoothing)
    machine = GestureStateMachine()

    fps_window = deque(maxlen=30)
    last_t = time.monotonic()
    last_state = None

    quit_flag = {"stop": False}
    if args.headless:
        try:
            import keyboard as _kb
            _kb.add_hotkey("ctrl+shift+q", lambda: quit_flag.update(stop=True))
            print("[handvol] headless mode — press Ctrl+Shift+Q to quit")
        except Exception as e:
            print(f"[handvol] headless quit hotkey unavailable ({e}); kill via Task Manager")

    with GestureSource(cam_index=args.cam) as source:
        while True:
            frame, latest = source.read()
            if frame is None:
                break

            gesture, score, landmarks = (latest if latest else ("None", 0.0, None))
            event = machine.step(gesture)

            # --- React to state machine events ---
            if event is Event.ENTER_SCRUB and landmarks is not None:
                tip_y = landmarks[INDEX_TIP].y
                current_vol = 50.0 if args.no_audio else audio.get_volume()
                scrubber.enter(tip_y, current_vol)

            elif event is Event.UPDATE_SCRUB and landmarks is not None and scrubber.active:
                tip_y = landmarks[INDEX_TIP].y
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

            # FPS (always tracked, for logging)
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

            # --- Overlay (skipped in headless to save CPU) ---
            if not args.headless:
                if landmarks is not None:
                    draw_landmarks(frame, landmarks)
                draw_state(frame, machine.state.value)
                draw_gesture(frame, gesture, score)
                draw_volume(frame, vol_now)
                if machine.state is State.SCRUB and scrubber.active and landmarks is not None:
                    tip = landmarks[INDEX_TIP]
                    draw_scrub_indicator(frame, scrubber.anchor_y, tip.y, tip.x)
                draw_fps(frame, fps)
                cv2.imshow("HandVol", frame)

            # --- Logging ---
            if machine.state != last_state or event is not Event.NONE:
                if args.debug or event in (
                    Event.ENTER_SCRUB, Event.EXIT_SCRUB,
                    Event.TOGGLE_MUTE, Event.TOGGLE_PLAYPAUSE,
                ):
                    print(f"[{machine.state.value:14s}] gesture={gesture:14s} "
                          f"event={event.value:18s} fps={fps:5.1f}")
                last_state = machine.state

            if args.debug and event is Event.UPDATE_SCRUB and scrubber.active:
                print(f"  scrub anchor_y={scrubber.anchor_y:.3f} "
                      f"smoothed_y={scrubber.smoothed_y:.3f} "
                      f"vol={vol_now if vol_now is not None else 'n/a'}")

            if args.headless:
                if quit_flag["stop"]:
                    break
            else:
                if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
                    break

    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
