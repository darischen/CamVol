"""MediaPipe Face Landmarker wrapper + bbox helper.

The embedder runs in LIVE_STREAM mode, mirroring the gesture recognizer
pattern in capture.py.
"""
import threading
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


FACE_MODEL_FILENAME = "face_landmarker.task"

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / FACE_MODEL_FILENAME
)
MAX_FACES = 1  # Only the most prominent face is needed; MediaPipe picks it.

# Coalescing rate-limit on face landmarker submissions. The 478-point
# mesh inference is heavy enough that submitting at full camera FPS
# (30 Hz) can saturate it on modest CPUs and back the main loop down to
# ~15 FPS through scheduler back-pressure. 10 Hz is plenty for the dots
# overlay and bbox refresh; identity is checked even less often (~3 Hz
# in IdentityEncoder), so the bbox handoff stays fresh enough.
FACE_EMBED_MIN_INTERVAL_MS = 100


def landmarks_to_bbox(face_landmarks, frame_shape):
    """Compute the pixel bbox enclosing a MediaPipe face landmark list.

    `face_landmarks` is a list of NormalizedLandmark objects (478 entries
    from the Face Landmarker). `frame_shape` is `(h, w)` or `(h, w, c)`.

    Returns `(top, right, bottom, left)` in pixel coords — the format
    `face_recognition.face_encodings`' `known_face_locations` parameter
    expects. Returns None for empty input.
    """
    if not face_landmarks:
        return None
    h = frame_shape[0]
    w = frame_shape[1]
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    left = max(0, int(min(xs) * w))
    right = min(w, int(max(xs) * w))
    top = max(0, int(min(ys) * h))
    bottom = min(h, int(max(ys) * h))
    return (top, right, bottom, left)


class FaceEmbedder:
    """MediaPipe Face Landmarker in LIVE_STREAM mode.

    Submit frames with `submit(mp_image, ts_ms)`; the latest list of
    embeddings (one per detected face, up to MAX_FACES) is available via
    `latest()`. Mirrors the GestureSource async pattern in capture.py.
    """

    def __init__(self, model_path=None, min_interval_ms=FACE_EMBED_MIN_INTERVAL_MS):
        self.model_path = str(model_path or _DEFAULT_MODEL_PATH)
        self._lock = threading.Lock()
        self._latest_face_landmarks: list = []  # list[list[NormalizedLandmark]]
        self._latest_ts_ns = 0
        self._landmarker = None
        # Wall-clock throttle: drop submit() calls closer than this to
        # the previous accepted call. Coalescing — never queues.
        self._min_interval_s = min_interval_ms / 1000.0
        self._last_submit_t = 0.0

    def open(self) -> None:
        base_opts = mp_python.BaseOptions(model_asset_path=self.model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=MAX_FACES,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def submit(self, mp_image, ts_ms: int) -> None:
        if self._landmarker is None:
            return
        now = time.monotonic()
        if now - self._last_submit_t < self._min_interval_s:
            return  # coalesce — face mesh inference is too slow for per-frame
        self._last_submit_t = now
        self._landmarker.detect_async(mp_image, ts_ms)

    def latest(self):
        """Return (face_landmarks_list, ts_ns).

        face_landmarks_list is a list of lists of NormalizedLandmark, one
        inner list per detected face. Empty when no face was detected in
        the most recent frame.
        """
        with self._lock:
            return list(self._latest_face_landmarks), self._latest_ts_ns

    def _on_result(self, result, output_image, timestamp_ms):
        face_landmarks_list: list = []
        if result.face_landmarks:
            face_landmarks_list = list(result.face_landmarks)
        with self._lock:
            self._latest_face_landmarks = face_landmarks_list
            self._latest_ts_ns = time.monotonic_ns()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
