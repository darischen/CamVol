import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"


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

    def _on_result(self, result, output_image, timestamp_ms):
        gesture_name = "None"
        score = 0.0
        landmarks = None
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            gesture_name = top.category_name or "None"
            score = top.score
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
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
        self._start_ns = time.monotonic_ns()

    def read(self):
        """Grab a frame, mirror it, submit to recognizer. Returns (frame, latest_result)."""
        ok, frame = self._cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
        self._recognizer.recognize_async(mp_image, ts_ms)

        with self._lock:
            latest = self._latest
        return frame, latest

    def close(self):
        if self._recognizer is not None:
            self._recognizer.close()
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
