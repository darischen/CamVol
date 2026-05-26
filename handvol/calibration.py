"""Standalone face calibration flow.

Run from the command line:
    python -m handvol.calibration
Or via the tray menu (handvol.pyw spawns this in a subprocess).

The user is walked through a fixed list of poses. For each pose:
  * a countdown gives them time to settle into the pose;
  * frames are pulled from the camera; the first frame where the face
    embedder produces a valid embedding is accepted as that pose's
    capture;
  * if no embedding lands within a per-pose timeout, the pose is retried.
After all poses are captured the resulting FaceProfile is written to disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp

from handvol.face_detect import FaceEmbedder
from handvol.face_profile import FaceProfile, DEFAULT_PROFILE_PATH
from handvol.overlay import draw_face_landmarks


# Each pose: short label + instruction text shown to the user.
# Order is chosen so motion between consecutive poses is small.
POSES = [
    ("center_close",      "CENTER - close to camera"),
    ("center_medium",     "CENTER - normal distance"),
    ("center_far",        "CENTER - lean back / farther away"),
    ("up",                "Look UP"),
    ("up_left",           "Look UP-LEFT"),
    ("left",              "Look LEFT"),
    ("down_left",         "Look DOWN-LEFT"),
    ("down",              "Look DOWN"),
    ("down_right",        "Look DOWN-RIGHT"),
    ("right",             "Look RIGHT"),
    ("up_right",          "Look UP-RIGHT"),
    ("center_neutral_1",  "Return to CENTER, neutral expression"),
    ("profile_left",      "Turn head LEFT (show right cheek/ear) - remove over-ear headphones if any"),
    ("profile_right",     "Turn head RIGHT (show left cheek/ear) - remove over-ear headphones if any"),
    ("tilt_left",         "Tilt head LEFT (ear toward shoulder)"),
    ("tilt_right",        "Tilt head RIGHT (ear toward shoulder)"),
    ("center_neutral_2",  "CENTER again, slightly closer"),
    ("center_neutral_3",  "CENTER again, slightly farther"),
    ("chin_up",           "Chin UP (look slightly above camera)"),
    ("chin_down",         "Chin DOWN (look slightly below camera)"),
]

COUNTDOWN_SECONDS = 2.0
PER_POSE_TIMEOUT_SECONDS = 8.0
WINDOW_TITLE = "HandVol - Face Calibration"


@dataclass
class CalibrationResult:
    profile: FaceProfile | None
    completed_pose_count: int
    aborted: bool


def _draw_text(frame, text, y, color=(255, 255, 255), scale=0.8, thickness=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_pose_screen(frame, label, instruction, idx, total, status_text, status_color):
    h = frame.shape[0]
    _draw_text(frame, f"Pose {idx + 1}/{total}: {label}", 36, (80, 220, 240))
    _draw_text(frame, instruction, 72, (255, 255, 255), 0.7, 2)
    _draw_text(frame, status_text, h - 24, status_color, 0.7, 2)
    _draw_text(frame, "Press Q to abort", h - 56, (160, 160, 160), 0.5, 1)


def _capture_pose(cap, embedder, label, instruction, idx, total):
    """Run countdown then collect a single embedding for this pose.

    Returns the embedding, or None if the user pressed Q.
    Retries forever within the timeout if no face is detected.
    """
    countdown_start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic_ns() // 1_000_000))
        embedder.submit(mp_image, ts_ms)

        # Always pull the latest result so we can overlay the dots while
        # the user is settling into the pose.
        embs, face_lms, _ = embedder.latest()
        if face_lms:
            draw_face_landmarks(frame, face_lms)

        elapsed = time.monotonic() - countdown_start

        if elapsed < COUNTDOWN_SECONDS:
            remaining = COUNTDOWN_SECONDS - elapsed
            _draw_pose_screen(
                frame, label, instruction, idx, len(POSES),
                f"Hold still... {remaining:.1f}", (60, 220, 240),
            )
        else:
            if embs:
                # During calibration the user is alone in frame; the first
                # detected face is the right one.
                emb = embs[0]
                _draw_pose_screen(
                    frame, label, instruction, idx, len(POSES),
                    "Captured!", (80, 220, 120),
                )
                cv2.imshow(WINDOW_TITLE, frame)
                cv2.waitKey(250)  # brief confirmation flash
                return emb
            else:
                # No face yet - keep trying until timeout, then restart countdown.
                if elapsed > COUNTDOWN_SECONDS + PER_POSE_TIMEOUT_SECONDS:
                    countdown_start = time.monotonic()  # restart pose
                    continue
                _draw_pose_screen(
                    frame, label, instruction, idx, len(POSES),
                    "No face detected - adjust position", (80, 80, 240),
                )

        cv2.imshow(WINDOW_TITLE, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            return None


def run_calibration(cam_index: int = 0, output_path: Path = DEFAULT_PROFILE_PATH) -> CalibrationResult:
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # We don't know the embedding dim until we get the first capture, so
    # build the profile lazily.
    profile: FaceProfile | None = None

    try:
        with FaceEmbedder() as embedder:
            for idx, (label, instruction) in enumerate(POSES):
                emb = _capture_pose(cap, embedder, label, instruction, idx, len(POSES))
                if emb is None:
                    return CalibrationResult(profile=None, completed_pose_count=idx, aborted=True)
                if profile is None:
                    profile = FaceProfile.create_empty(embedding_dim=emb.shape[0])
                profile.add_capture(emb)
            # End of pose loop.
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    assert profile is not None
    profile.save(output_path)
    return CalibrationResult(profile=profile, completed_pose_count=len(POSES), aborted=False)


def main():
    p = argparse.ArgumentParser(description="HandVol face calibration")
    p.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH,
                   help="Where to write the face profile (default: data/face_profile.npz)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing profile without prompting")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        ans = input(f"Profile already exists at {args.output}. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 1

    result = run_calibration(cam_index=args.cam, output_path=args.output)
    if result.aborted:
        print(f"Calibration aborted after {result.completed_pose_count} pose(s). No profile written.")
        return 1
    print(f"Saved face profile with {result.profile.capture_count} captures to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
