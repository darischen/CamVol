"""MediaPipe Face Landmarker wrapper + landmark-to-embedding helper.

The embedder runs in LIVE_STREAM mode, mirroring the gesture recognizer
pattern in capture.py. The embedding helper is intentionally a pure
function so it can be unit-tested without the model or a camera.
"""
import threading
import time
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


EXPECTED_LANDMARK_COUNT = 478  # MediaPipe Face Landmarker output
FACE_MODEL_FILENAME = "face_landmarker.task"

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / FACE_MODEL_FILENAME
)
MAX_FACES = 3  # Spec: if any face in frame matches, allow gestures.


def landmarks_to_embedding(landmarks):
    """Convert face landmarks to a translation+scale-invariant identity vector.

    Steps:
      1. Stack the (x, y, z) of each NormalizedLandmark into an (N, 3) array.
      2. Subtract the centroid so the embedding is invariant to where the
         face is located in the frame.
      3. Divide by the RMS distance from the centroid so it is invariant to
         how close the face is to the camera.
      4. Flatten to a 1-D vector; cosine similarity on this vector compares
         relative facial geometry, which is the part that is identity-bearing.

    Returns None if the input is missing or has fewer landmarks than the
    Face Landmarker is expected to emit.
    """
    if not landmarks or len(landmarks) < EXPECTED_LANDMARK_COUNT:
        return None
    pts = np.asarray(
        [(lm.x, lm.y, lm.z) for lm in landmarks[:EXPECTED_LANDMARK_COUNT]],
        dtype=np.float32,
    )
    pts -= pts.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((pts ** 2).sum() / pts.shape[0]))
    if scale < 1e-9:
        return None
    pts /= scale
    return pts.reshape(-1)


class FaceEmbedder:
    """MediaPipe Face Landmarker in LIVE_STREAM mode.

    Submit frames with `submit(mp_image, ts_ms)`; the latest list of
    embeddings (one per detected face, up to MAX_FACES) is available via
    `latest()`. Mirrors the GestureSource async pattern in capture.py.
    """

    def __init__(self, model_path=None):
        self.model_path = str(model_path or _DEFAULT_MODEL_PATH)
        self._lock = threading.Lock()
        self._latest_embeddings: list = []  # list[np.ndarray]
        self._latest_ts_ns = 0
        self._landmarker = None

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
        self._landmarker.detect_async(mp_image, ts_ms)

    def latest(self):
        """Return (embeddings_list, ts_ns) of the most recent result.

        embeddings_list is a list of embeddings (numpy arrays). Empty if no
        face was detected in the most recent frame.
        """
        with self._lock:
            return list(self._latest_embeddings), self._latest_ts_ns

    def _on_result(self, result, output_image, timestamp_ms):
        embeddings: list = []
        if result.face_landmarks:
            for face in result.face_landmarks:
                emb = landmarks_to_embedding(face)
                if emb is not None:
                    embeddings.append(emb)
        with self._lock:
            self._latest_embeddings = embeddings
            self._latest_ts_ns = time.monotonic_ns()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
