import math
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from handvol.face_detect import FaceEmbedder


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"

OK_SIGN_PINCH_THRESHOLD = 0.06

SIDE_THUMB_HORIZ_RATIO = 1.2   # |dx| must exceed |dy| by this factor
SIDE_THUMB_EXTEND_RATIO = 1.3  # tip-to-wrist must exceed cmc-to-wrist by this factor
SIDE_THUMB_CURL_RATIO = 1.2    # fingertip-to-wrist must stay within this * mcp-to-wrist


def _detect_side_thumb(landmarks, handedness):
    """Return a label like 'left_hand_thumb_right' or None.

    Fires when the thumb sticks out roughly horizontally with the other
    four fingers curled into a fist. Direction is reported in the user's
    real-world frame: thumb tip pointing toward the user's right is
    '..._thumb_right' for either hand.
    """
    if not landmarks or len(landmarks) < 21 or handedness not in ("Left", "Right"):
        return None

    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    thumb_tip = landmarks[4]

    dx = thumb_tip.x - thumb_cmc.x
    dy = thumb_tip.y - thumb_cmc.y
    if abs(dx) < abs(dy) * SIDE_THUMB_HORIZ_RATIO:
        return None

    tip_dist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    cmc_dist = math.hypot(thumb_cmc.x - wrist.x, thumb_cmc.y - wrist.y)
    if tip_dist < cmc_dist * SIDE_THUMB_EXTEND_RATIO:
        return None

    for mcp_idx, tip_idx in ((5, 8), (9, 12), (13, 16), (17, 20)):
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        mcp_to_wrist = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
        tip_to_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        if tip_to_wrist > mcp_to_wrist * SIDE_THUMB_CURL_RATIO:
            return None

    hand_prefix = "right_hand" if handedness == "Left" else "left_hand"
    direction = "right" if dx > 0 else "left"
    return f"{hand_prefix}_thumb_{direction}"


def _detect_ok_sign(landmarks):
    """Return True if the 21 hand landmarks form an OK sign.

    Thumb tip touches index tip; middle, ring, pinky extended upward.
    """
    if not landmarks or len(landmarks) < 21:
        return False
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    pinch = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    if pinch >= OK_SIGN_PINCH_THRESHOLD:
        return False
    middle_extended = landmarks[12].y < landmarks[10].y
    ring_extended = landmarks[16].y < landmarks[14].y
    pinky_extended = landmarks[20].y < landmarks[18].y
    return middle_extended and ring_extended and pinky_extended


class GestureSource:
    """Webcam + MediaPipe GestureRecognizer in LIVE_STREAM mode.

    Frames are pushed in synchronously; results arrive on a worker thread
    via callback and are stashed in a single latest-result slot.
    """

    def __init__(self, cam_index=0, width=640, height=480, model_path=None):
        self.width = width
        self.height = height
        self.cam_index = cam_index
        self.model_path = str(model_path or MODEL_PATH)

        self._cap = None
        self._recognizer = None
        self._lock = threading.Lock()
        self._latest = None  # (gesture_name, score, landmarks)
        self._start_ns = None
        self._embedder = FaceEmbedder()

    def _on_result(self, result, output_image, timestamp_ms):
        gesture_name = "None"
        score = 0.0
        landmarks = None
        handedness = None
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            gesture_name = top.category_name or "None"
            score = top.score
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            if _detect_ok_sign(landmarks):
                gesture_name = "OK_sign"
                score = 1.0
            else:
                side = _detect_side_thumb(landmarks, handedness)
                if side is not None:
                    gesture_name = side
                    score = 1.0
        with self._lock:
            self._latest = (gesture_name, score, landmarks)

    def open(self):
        self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.cam_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        base_opts = mp_python.BaseOptions(model_asset_path=self.model_path)
        opts = mp_vision.GestureRecognizerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self._on_result,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(opts)
        self._embedder.open()
        self._start_ns = time.monotonic_ns()

    def read(self):
        """Grab a frame, mirror it, submit to recognizer + face embedder.
        Returns (frame, latest_result) where latest_result is
        (gesture_name, score, landmarks, face_embeddings) or None.
        face_embeddings is a list (possibly empty) of face identity vectors.
        """
        ok, frame = self._cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
        self._recognizer.recognize_async(mp_image, ts_ms)
        self._embedder.submit(mp_image, ts_ms)

        with self._lock:
            latest = self._latest
        face_embs, _ = self._embedder.latest()
        if latest is None:
            return frame, None
        gesture_name, score, landmarks = latest
        return frame, (gesture_name, score, landmarks, face_embs)

    def close(self):
        self._embedder.close()
        if self._recognizer is not None:
            self._recognizer.close()
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
