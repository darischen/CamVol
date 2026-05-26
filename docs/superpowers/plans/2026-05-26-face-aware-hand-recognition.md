# Face-Aware Hand Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate HandVol's gesture event dispatch behind face recognition so that only the calibrated user's face unlocks gesture control.

**Architecture:** Add a second MediaPipe `LIVE_STREAM` task (Face Landmarker) to the existing capture pipeline. Convert detected face landmarks into a translation/scale-normalized identity embedding, compare against a stored profile of ~20 calibration captures via cosine similarity, and gate the main-loop event dispatch on the match result. Calibration is a separate process launched from the tray menu (or CLI).

**Tech Stack:** MediaPipe Tasks (`mediapipe.tasks.python.vision.FaceLandmarker`), numpy (cosine similarity, `.npz` profile file), OpenCV (calibration UI), pystray (tray menu), existing handvol modules.

**Spec:** `docs/superpowers/specs/2026-05-26-face-aware-hand-recognition-design.md`

---

## File Structure

**Create:**
- `handvol/face_profile.py` — `FaceProfile` class: embedding storage, save/load, `matches()`.
- `handvol/face_detect.py` — `FaceEmbedder` wrapper around MediaPipe Face Landmarker; `landmarks_to_embedding()` helper.
- `handvol/calibration.py` — Standalone calibration flow + `__main__` entry point.
- `tests/test_face_profile.py` — Unit tests for `FaceProfile`.
- `tests/test_face_embedding.py` — Unit tests for the `landmarks_to_embedding()` pure-math helper.
- `data/` (directory only; contents gitignored).

**Modify:**
- `.gitignore` — Add `data/`.
- `handvol/capture.py` — Add face embedder task, return embedding in result tuple.
- `handvol/overlay.py` — Add `draw_lock_state()` helper.
- `handvol.pyw` — Load profile, dispatch gating, "Calibrate face..." tray menu item.
- `README.md` — Mention face calibration step and model download.

**Constants live with the module that uses them** (mirrors the existing `OK_SIGN_PINCH_THRESHOLD` pattern in `capture.py`):
- `face_profile.py`: `MATCH_THRESHOLD = 0.92` (cosine similarity on landmark embeddings is high-magnitude; tune in Task 11).
- `face_detect.py`: `FACE_MODEL_FILENAME = "face_landmarker.task"`.
- `capture.py`: `NO_FACE_GRACE_FRAMES = 15`.

---

## Task 1: Add data/ to .gitignore and download face landmarker model

**Files:**
- Modify: `.gitignore`
- Create: `data/.gitkeep` (so the directory exists in fresh clones; gitignore keeps contents out)
- Modify: `README.md` (add model download step)

- [ ] **Step 1: Add `data/` to `.gitignore`**

Edit `.gitignore`. After the existing `models/*.task` line, add:

```
data/
!data/.gitkeep
```

The negation `!data/.gitkeep` lets us commit an empty marker file so the directory exists for new clones, but everything else inside `data/` (the face profile) stays out of git.

- [ ] **Step 2: Create the marker file**

```bash
mkdir -p data
touch data/.gitkeep
```

- [ ] **Step 3: Download the MediaPipe face landmarker model**

```bash
mkdir -p models
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Expected: file `models/face_landmarker.task` of ~3.7 MB. Already covered by `models/*.task` in `.gitignore` so it will not be committed.

- [ ] **Step 4: Update README**

Add a new section after the existing "Download the MediaPipe gesture model bundle" block:

````markdown
Also download the face landmarker model:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Save it as `models/face_landmarker.task`.

After first launch, run the face calibration once via the tray icon's
**Calibrate face...** menu item (or `python -m handvol.calibration`).
Gestures are blocked until calibration has been completed.
````

- [ ] **Step 5: Verify**

```bash
git status
```
Expected: `.gitignore`, `data/.gitkeep`, and `README.md` show as changes; `models/face_landmarker.task` and any future `data/face_profile.npz` do NOT appear.

- [ ] **Step 6: Commit**

```bash
git add .gitignore data/.gitkeep README.md
git commit -m "chore: gitignore data/, document face landmarker model"
```

---

## Task 2: Embedding helper — convert face landmarks to identity vector

**Files:**
- Create: `handvol/face_detect.py` (helper only this task; the embedder class comes in Task 4)
- Create: `tests/test_face_embedding.py`

**Why a pure helper first:** The math (subtract centroid, divide by interocular distance, flatten) is unit-testable without MediaPipe or a camera. Isolating it makes the embedder wrapper trivial.

- [ ] **Step 1: Write failing tests**

Create `tests/test_face_embedding.py`:

```python
import math

import numpy as np
import pytest

from handvol.face_detect import landmarks_to_embedding


class _LM:
    """Minimal stand-in for MediaPipe's NormalizedLandmark."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks(n=478, scale=1.0, offset=(0.0, 0.0)):
    """Generate n landmarks on a deterministic grid for testing."""
    rng = np.random.default_rng(seed=0)
    pts = rng.uniform(-0.5, 0.5, size=(n, 3)) * scale
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    return [_LM(p[0], p[1], p[2]) for p in pts]


def test_returns_none_for_missing_landmarks():
    assert landmarks_to_embedding(None) is None
    assert landmarks_to_embedding([]) is None


def test_returns_none_for_too_few_landmarks():
    assert landmarks_to_embedding(_make_landmarks(n=10)) is None


def test_embedding_shape_is_flat_vector():
    lms = _make_landmarks(n=478)
    emb = landmarks_to_embedding(lms)
    assert emb is not None
    assert emb.ndim == 1
    assert emb.shape[0] == 478 * 3


def test_embedding_is_translation_invariant():
    base = _make_landmarks(n=478, offset=(0.0, 0.0))
    shifted = _make_landmarks(n=478, offset=(0.2, -0.1))  # same rng seed -> same pattern
    e1 = landmarks_to_embedding(base)
    e2 = landmarks_to_embedding(shifted)
    # Cosine similarity should be ~1.0 after centroid removal.
    cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_embedding_is_scale_invariant():
    base = _make_landmarks(n=478, scale=1.0)
    bigger = _make_landmarks(n=478, scale=2.5)  # same rng seed -> same pattern, scaled
    e1 = landmarks_to_embedding(base)
    e2 = landmarks_to_embedding(bigger)
    cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    assert cos == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_face_embedding.py -v
```
Expected: ImportError / ModuleNotFoundError, since `handvol/face_detect.py` does not exist yet.

- [ ] **Step 3: Implement `landmarks_to_embedding`**

Create `handvol/face_detect.py`:

```python
"""MediaPipe Face Landmarker wrapper + landmark-to-embedding helper.

The embedder runs in LIVE_STREAM mode, mirroring the gesture recognizer
pattern in capture.py. The embedding helper is intentionally a pure
function so it can be unit-tested without the model or a camera.
"""
from pathlib import Path

import numpy as np


EXPECTED_LANDMARK_COUNT = 478  # MediaPipe Face Landmarker output
FACE_MODEL_FILENAME = "face_landmarker.task"


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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_face_embedding.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add handvol/face_detect.py tests/test_face_embedding.py
git commit -m "feat(face): add landmarks_to_embedding helper with unit tests"
```

---

## Task 3: `FaceProfile` — storage, save/load, match

**Files:**
- Create: `handvol/face_profile.py`
- Create: `tests/test_face_profile.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_face_profile.py`:

```python
import numpy as np
import pytest

from handvol.face_profile import FaceProfile, MATCH_THRESHOLD


def _unit(vec):
    return vec / np.linalg.norm(vec)


def test_empty_profile_does_not_match():
    p = FaceProfile.create_empty(embedding_dim=8)
    is_match, sim = p.matches(_unit(np.ones(8, dtype=np.float32)))
    assert is_match is False
    assert sim == 0.0


def test_identical_embedding_matches_at_similarity_1():
    p = FaceProfile.create_empty(embedding_dim=8)
    v = _unit(np.arange(1, 9, dtype=np.float32))
    p.add_capture(v)
    is_match, sim = p.matches(v)
    assert is_match is True
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_embedding_does_not_match():
    p = FaceProfile.create_empty(embedding_dim=8)
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    p.add_capture(a)
    is_match, sim = p.matches(b)
    assert is_match is False
    assert sim == pytest.approx(0.0, abs=1e-6)


def test_max_similarity_used_across_multiple_captures():
    p = FaceProfile.create_empty(embedding_dim=8)
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[3] = 1.0
    p.add_capture(a)
    p.add_capture(b)
    # query is close to b, far from a -> max similarity should be ~b's
    query = b.copy()
    _, sim = p.matches(query)
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_threshold_boundary(tmp_path):
    p = FaceProfile.create_empty(embedding_dim=8)
    v = _unit(np.arange(1, 9, dtype=np.float32))
    p.add_capture(v)
    # Slightly perturbed query: stays well above threshold for similar faces.
    perturbed = _unit(v + 0.01 * np.ones(8, dtype=np.float32))
    is_match, sim = p.matches(perturbed)
    assert is_match is (sim >= MATCH_THRESHOLD)


def test_save_and_load_round_trip(tmp_path):
    p = FaceProfile.create_empty(embedding_dim=8)
    v1 = _unit(np.arange(1, 9, dtype=np.float32))
    v2 = _unit(np.arange(8, 0, -1).astype(np.float32))
    p.add_capture(v1)
    p.add_capture(v2)

    path = tmp_path / "profile.npz"
    p.save(path)
    assert path.exists()

    loaded = FaceProfile.load(path)
    assert loaded is not None
    assert loaded.embeddings.shape == (2, 8)
    np.testing.assert_allclose(loaded.embeddings[0], v1, rtol=1e-6)
    np.testing.assert_allclose(loaded.embeddings[1], v2, rtol=1e-6)


def test_load_missing_file_returns_none(tmp_path):
    assert FaceProfile.load(tmp_path / "nope.npz") is None


def test_load_corrupted_file_returns_none(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a real npz file")
    assert FaceProfile.load(bad) is None


def test_add_capture_rejects_wrong_dim():
    p = FaceProfile.create_empty(embedding_dim=8)
    with pytest.raises(ValueError):
        p.add_capture(np.zeros(7, dtype=np.float32))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_face_profile.py -v
```
Expected: ImportError, since `handvol/face_profile.py` does not exist.

- [ ] **Step 3: Implement `FaceProfile`**

Create `handvol/face_profile.py`:

```python
"""On-disk face identity profile.

Stores N L2-normalized embeddings produced by `face_detect.landmarks_to_embedding`.
`matches()` returns the maximum cosine similarity across all stored
captures, plus a boolean against `MATCH_THRESHOLD`. Storage is a single
`.npz` file under data/face_profile.npz (gitignored).

The threshold is high because cosine similarity on translation+scale
normalized landmark vectors of the same identity tends to sit well above
0.9; cross-identity similarity drops sharply. Tune empirically in Task 11.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MATCH_THRESHOLD = 0.92
PROFILE_VERSION = 1
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "face_profile.npz"

_log = logging.getLogger(__name__)


class FaceProfile:
    def __init__(self, embeddings: np.ndarray, created_at: str = "", version: int = PROFILE_VERSION):
        # embeddings is (N, D), float32, each row L2-normalized.
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.created_at = created_at
        self.version = version

    @classmethod
    def create_empty(cls, embedding_dim: int) -> "FaceProfile":
        empty = np.zeros((0, embedding_dim), dtype=np.float32)
        return cls(empty, created_at="", version=PROFILE_VERSION)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def capture_count(self) -> int:
        return int(self.embeddings.shape[0])

    def add_capture(self, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1 or emb.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding of shape ({self.embedding_dim},), "
                f"got {emb.shape}"
            )
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            raise ValueError("Cannot add a zero-norm embedding")
        emb_unit = emb / norm
        self.embeddings = np.vstack([self.embeddings, emb_unit[None, :]])

    def matches(self, embedding: np.ndarray) -> tuple[bool, float]:
        if self.capture_count == 0:
            return False, 0.0
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            return False, 0.0
        emb_unit = emb / norm
        # Stored embeddings are already unit; cosine sim is just dot product.
        sims = self.embeddings @ emb_unit
        max_sim = float(sims.max())
        return (max_sim >= MATCH_THRESHOLD), max_sim

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings,
            created_at=np.array(
                self.created_at or datetime.now(timezone.utc).isoformat()
            ),
            version=np.array(self.version),
        )

    @classmethod
    def load(cls, path: Path) -> "FaceProfile | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                embeddings = data["embeddings"]
                created_at = str(data["created_at"]) if "created_at" in data else ""
                version = int(data["version"]) if "version" in data else PROFILE_VERSION
        except (OSError, ValueError, KeyError, EOFError) as exc:
            _log.warning("Failed to load face profile at %s: %s", path, exc)
            return None
        if embeddings.ndim != 2:
            _log.warning("Face profile at %s has unexpected shape %s", path, embeddings.shape)
            return None
        return cls(embeddings.astype(np.float32, copy=False), created_at=created_at, version=version)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_face_profile.py -v
```
Expected: all 9 tests pass.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
python -m pytest tests/ -q
```
Expected: all existing scrubber tests still pass + the two new test files pass.

- [ ] **Step 6: Commit**

```bash
git add handvol/face_profile.py tests/test_face_profile.py
git commit -m "feat(face): FaceProfile with cosine-similarity matching and npz round-trip"
```

---

## Task 4: `FaceEmbedder` — async MediaPipe wrapper

**Files:**
- Modify: `handvol/face_detect.py` (append class; helper from Task 2 stays)

**Why minimal tests here:** Driving the real MediaPipe model from unit tests would require a real face image and the model bundle. Manual integration coverage happens during calibration testing in Task 11. The class is small and mirrors a pattern already proven in `capture.py`.

- [ ] **Step 1: Extend `handvol/face_detect.py` with the embedder class**

Append to `handvol/face_detect.py` (below `landmarks_to_embedding`):

```python
import threading
import time

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / FACE_MODEL_FILENAME
)


MAX_FACES = 3  # Spec: if any face in frame matches, allow gestures.


class FaceEmbedder:
    """MediaPipe Face Landmarker in LIVE_STREAM mode.

    Submit frames with `submit(rgb_frame, ts_ms)`; the latest list of
    embeddings (one per detected face, up to MAX_FACES) is available via
    `latest()`. Mirrors the GestureSource async pattern in capture.py.
    """

    def __init__(self, model_path: Path | str | None = None):
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

    def submit(self, mp_image: "mp.Image", ts_ms: int) -> None:
        if self._landmarker is None:
            return
        self._landmarker.detect_async(mp_image, ts_ms)

    def latest(self):
        """Return (embeddings_list, ts_ns) of the most recent result.

        embeddings_list is a list of L2-input embeddings — empty if no
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
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from handvol.face_detect import FaceEmbedder, landmarks_to_embedding; print('ok')"
```
Expected: prints `ok` with no traceback.

- [ ] **Step 3: Run the full test suite — no regressions**

```bash
python -m pytest tests/ -q
```
Expected: all tests still pass (no new tests added in this task).

- [ ] **Step 4: Commit**

```bash
git add handvol/face_detect.py
git commit -m "feat(face): FaceEmbedder LIVE_STREAM wrapper around MediaPipe Face Landmarker"
```

---

## Task 5: Integrate face embedder into `GestureSource`

**Files:**
- Modify: `handvol/capture.py`

The capture layer keeps its result-shape contract simple: it now returns `(gesture_name, score, landmarks, face_embedding)`. The recognition decision (`face_recognized` bool) belongs to the caller — that way `capture.py` knows nothing about `FaceProfile`.

- [ ] **Step 1: Modify `handvol/capture.py`**

At the top of the file, add the `FaceEmbedder` import alongside the existing imports:

```python
from handvol.face_detect import FaceEmbedder
```

In `GestureSource.__init__`, after the existing initialization lines, add the embedder attribute:

```python
        self._embedder = FaceEmbedder()
```

In `GestureSource.open()`, after `self._recognizer = mp_vision.GestureRecognizer.create_from_options(opts)`, add:

```python
        self._embedder.open()
```

In `GestureSource.close()`, add (before the cap release):

```python
        self._embedder.close()
```

In `GestureSource.read()`, modify the body so the face embedder receives the same frame and the result tuple grows:

```python
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
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from handvol.capture import GestureSource; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Run all unit tests — no regressions**

```bash
python -m pytest tests/ -q
```
Expected: all tests pass (capture.py has no unit tests of its own; we're verifying the import graph still loads).

- [ ] **Step 4: Commit**

```bash
git add handvol/capture.py
git commit -m "feat(face): wire face embedder into GestureSource read loop"
```

---

## Task 6: Lock-state overlay helper

**Files:**
- Modify: `handvol/overlay.py`

- [ ] **Step 1: Add `draw_lock_state` to `handvol/overlay.py`**

Append at the bottom of `handvol/overlay.py`:

```python
def draw_lock_state(frame, recognized, has_profile):
    """Small top-right indicator: 'UNLOCKED' (green), 'LOCKED' (red),
    or 'NO PROFILE' (gray) when calibration is missing.
    """
    if not has_profile:
        text = "NO PROFILE"
        color = GRAY
    elif recognized:
        text = "UNLOCKED"
        color = GREEN
    else:
        text = "LOCKED"
        color = RED
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    _put(frame, text, (w - tw - 12, 56), color, 0.6, 2)
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from handvol.overlay import draw_lock_state; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add handvol/overlay.py
git commit -m "feat(face): draw_lock_state overlay indicator"
```

---

## Task 7: Calibration flow

**Files:**
- Create: `handvol/calibration.py`

This is the most user-facing module. It opens a camera window, walks the user through ~20 poses, captures an embedding for each, and writes the profile to `data/face_profile.npz`.

- [ ] **Step 1: Create `handvol/calibration.py`**

```python
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

from handvol.face_detect import FaceEmbedder, landmarks_to_embedding
from handvol.face_profile import FaceProfile, DEFAULT_PROFILE_PATH


# Each pose: short label + instruction text shown to the user.
# Order is chosen so motion between consecutive poses is small.
POSES = [
    ("center_close",      "CENTER — close to camera"),
    ("center_medium",     "CENTER — normal distance"),
    ("center_far",        "CENTER — lean back / farther away"),
    ("up",                "Look UP"),
    ("up_left",           "Look UP-LEFT"),
    ("left",              "Look LEFT"),
    ("down_left",         "Look DOWN-LEFT"),
    ("down",              "Look DOWN"),
    ("down_right",        "Look DOWN-RIGHT"),
    ("right",             "Look RIGHT"),
    ("up_right",          "Look UP-RIGHT"),
    ("center_neutral_1",  "Return to CENTER, neutral expression"),
    ("profile_left",      "Turn head LEFT (show right cheek/ear) — remove over-ear headphones if any"),
    ("profile_right",     "Turn head RIGHT (show left cheek/ear) — remove over-ear headphones if any"),
    ("tilt_left",         "Tilt head LEFT (ear toward shoulder)"),
    ("tilt_right",        "Tilt head RIGHT (ear toward shoulder)"),
    ("center_neutral_2",  "CENTER again, slightly closer"),
    ("center_neutral_3",  "CENTER again, slightly farther"),
    ("chin_up",           "Chin UP (look slightly above camera)"),
    ("chin_down",         "Chin DOWN (look slightly below camera)"),
]

COUNTDOWN_SECONDS = 2.0
PER_POSE_TIMEOUT_SECONDS = 8.0
WINDOW_TITLE = "HandVol — Face Calibration"


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


def _capture_pose(cap, embedder, label, instruction, idx, total) -> "np.ndarray | None":
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

        elapsed = time.monotonic() - countdown_start

        if elapsed < COUNTDOWN_SECONDS:
            remaining = COUNTDOWN_SECONDS - elapsed
            _draw_pose_screen(
                frame, label, instruction, idx, len(POSES),
                f"Hold still... {remaining:.1f}", (60, 220, 240),
            )
        else:
            embs, _ = embedder.latest()
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
                # No face yet — keep trying until timeout, then restart countdown.
                if elapsed > COUNTDOWN_SECONDS + PER_POSE_TIMEOUT_SECONDS:
                    countdown_start = time.monotonic()  # restart pose
                    continue
                _draw_pose_screen(
                    frame, label, instruction, idx, len(POSES),
                    "No face detected — adjust position", (80, 80, 240),
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
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from handvol.calibration import run_calibration, POSES; print(len(POSES))"
```
Expected: prints `20`.

- [ ] **Step 3: Run the full test suite — no regressions**

```bash
python -m pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add handvol/calibration.py
git commit -m "feat(face): calibration CLI flow with ~20 poses across angles and distances"
```

---

## Task 8: Tray menu integration in `handvol.pyw`

**Files:**
- Modify: `handvol.pyw`

Three things change in the entry point:
1. Load the face profile on startup and pass it into the capture loop.
2. Add the **Calibrate face...** tray menu item.
3. Gate event dispatch and render the lock indicator.

This task does all three together because they share state (the loaded profile, the pause-during-calibration flag).

- [ ] **Step 1: Add imports to `handvol.pyw`**

Near the existing `from handvol...` block at the top:

```python
import os
import subprocess
import sys

from handvol.face_detect import landmarks_to_embedding
from handvol.face_profile import FaceProfile, DEFAULT_PROFILE_PATH
```

- [ ] **Step 2: Add a profile holder + grace-frame counter to the worker state**

Inside `main()`, just before `paused = {"v": False}`, add:

```python
    # Mutable holder so on_calibrate can swap in a freshly loaded profile.
    profile_state = {"profile": FaceProfile.load(DEFAULT_PROFILE_PATH)}
```

- [ ] **Step 3: Modify `capture_loop` to accept and use the profile**

Replace the `capture_loop` function signature line and the result unpacking. Find:

```python
def capture_loop(args, show_evt, worker_stop, icon):
```

Replace with:

```python
def capture_loop(args, show_evt, worker_stop, icon, profile_state):
```

Inside `capture_loop`, replace the imports block at the top to include `draw_lock_state`:

```python
    from handvol.overlay import (
        draw_state, draw_gesture, draw_volume, draw_fps,
        draw_landmarks, draw_scrub_indicator, draw_lock_state,
    )
```

Right after `last_rendered_vol = None`, add:

```python
    NO_FACE_GRACE_FRAMES = 15
    no_face_streak = NO_FACE_GRACE_FRAMES  # start locked until proven otherwise
    last_recognized = False
```

Replace this line:

```python
            gesture, score, landmarks = (latest if latest else ("None", 0.0, None))
            event = machine.step(gesture)
```

With:

```python
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
```

Find the `if want_window:` block. After `draw_volume(frame, vol_now)`, add:

```python
                draw_lock_state(frame, recognized, has_profile=(profile_state["profile"] is not None and profile_state["profile"].capture_count > 0))
```

- [ ] **Step 4: Update `start_worker` to pass `profile_state`**

Find:

```python
    def start_worker():
        worker_state["stop"] = threading.Event()
        worker_state["thread"] = threading.Thread(
            target=capture_loop,
            args=(args, show_evt, worker_state["stop"], icon),
            daemon=True)
        worker_state["thread"].start()
```

Replace the `args=(...)` line with:

```python
            args=(args, show_evt, worker_state["stop"], icon, profile_state),
```

- [ ] **Step 5: Add `on_calibrate` callback and tray menu item**

Inside `main()`, after the `on_pause` definition and before `on_quit`, add:

```python
    def on_calibrate(icon, item):
        # Stop the capture worker so the camera is free, run calibration
        # in-process (blocks the tray callback thread, which is fine for
        # pystray), then reload the profile and restart the worker.
        stop_worker()
        try:
            python = sys.executable
            # If we were launched by pythonw, switch to python for the calibration
            # so OpenCV's imshow window is interactable.
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
```

Update the `menu = Menu(...)` block to add the new item:

```python
    menu = Menu(
        MenuItem("Show preview", on_toggle, default=True,
                 checked=lambda item: show_evt.is_set()),
        MenuItem("Pause", on_pause,
                 checked=lambda item: paused["v"]),
        MenuItem("Calibrate face...", on_calibrate),
        MenuItem("Quit", on_quit),
    )
```

- [ ] **Step 6: Run unit tests — no regressions**

```bash
python -m pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 7: Smoke-test the entry point's import graph**

```bash
python -c "import handvol; import importlib.util, runpy; spec = importlib.util.spec_from_file_location('hv', 'handvol.pyw'); m = importlib.util.module_from_spec(spec); print('ok')"
```
Expected: prints `ok`.

(Note: we don't `spec.loader.exec_module(m)` because `main()` would block on the tray; we just verify the file parses.)

- [ ] **Step 8: Commit**

```bash
git add handvol.pyw
git commit -m "feat(face): tray-menu calibration + face-gated gesture dispatch in handvol.pyw"
```

---

## Task 9: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CONTEXT.md` (if it exists)

- [ ] **Step 1: Add a "Face recognition" section to README**

After the existing gesture table in `README.md`, add:

```markdown
## Face recognition (gesture lock)

HandVol gates gesture control behind face recognition: only the
calibrated user's face will unlock gestures. With no profile, or when
the user's face is not visible, gestures are ignored and the preview
overlay shows **LOCKED** in the top right.

**First-run calibration:** open the tray menu (right-click the HandVol
icon) and choose **Calibrate face...**, or run:

```powershell
python -m handvol.calibration
```

You will be walked through ~20 short poses (looking up/down/left/right,
diagonals, profile views, two close/far variations). Hold each pose
neutral for ~2 seconds. Headphones are fine for most poses; you will be
prompted to briefly remove over-ear headphones for the two profile
captures so the ear/jawline is visible.

The face profile is stored at `data/face_profile.npz` and is **never**
committed to git (the `data/` directory is gitignored). Embeddings are
biometric data — keep the file local.
```

- [ ] **Step 2: Verify the file renders sanely**

```bash
head -80 README.md
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README section for face calibration + privacy note"
```

---

## Task 10: Update CONTEXT.md (if present)

**Files:**
- Modify: `CONTEXT.md`

- [ ] **Step 1: Check whether `CONTEXT.md` exists**

```bash
ls CONTEXT.md
```

If it does NOT exist, skip this task entirely and proceed to Task 11.

- [ ] **Step 2: Append a face-recognition section**

Add at the end of `CONTEXT.md`:

```markdown
## Face recognition gate

Gestures are dispatched only when the calibrated user's face is
recognized in the frame. Capture pipeline runs two MediaPipe tasks
concurrently in `LIVE_STREAM` mode: the existing gesture recognizer and
a face landmarker (`handvol/face_detect.py:FaceEmbedder`). Face
landmarks are converted to a translation+scale-normalized vector
(`landmarks_to_embedding`) and compared against ~20 stored captures via
max cosine similarity. The match threshold is `MATCH_THRESHOLD = 0.92`
in `handvol/face_profile.py`; tune empirically.

Lockout has a `NO_FACE_GRACE_FRAMES = 15` debouncer (~0.5s at 30fps)
so brief face-detection dropouts do not flash the lock state on and
off. While locked, the gesture stream is masked to `"None"`, which
lets the existing state machine in `handvol/state.py` exit any active
SCRUB cleanly without face-specific logic.

The profile is stored at `data/face_profile.npz` (gitignored).
```

- [ ] **Step 3: Commit**

```bash
git add CONTEXT.md
git commit -m "docs(context): note the face-recognition gate"
```

---

## Task 11: Manual end-to-end testing & threshold tuning

This task has no code changes — it is an explicit checklist for the implementer to walk through with the real camera. If any step fails, file the failure mode as a follow-up task rather than silently relaxing thresholds.

- [ ] **Step 1: Fresh state — verify the locked-out default**

Delete any existing profile:
```bash
rm -f data/face_profile.npz
```

Launch HandVol (`python handvol.pyw --show`). Expected behavior:
- Tray icon appears.
- Preview window shows **NO PROFILE** in the top right.
- Hand gestures (point, fist, etc.) produce **no** volume/playback changes.

- [ ] **Step 2: Calibration via tray menu**

Right-click the tray icon, choose **Calibrate face...**. Expected:
- The HandVol preview closes (worker is stopped).
- The calibration window opens.
- 20 poses run; each shows a 2s countdown and captures one embedding.
- After the last pose, the calibration window closes and the HandVol
  worker restarts automatically.
- `data/face_profile.npz` now exists.

- [ ] **Step 3: Verify unlocked behavior**

With only yourself in the frame:
- Preview overlay shows **UNLOCKED** in green.
- All existing gestures still work (point→scrub, fist→mute, palm→play/pause,
  victory→Spotify, OK sign→scrub, thumbs up→next, thumbs down→prev,
  ILoveYou→close Spotify).

- [ ] **Step 4: Verify lockout when face is occluded**

Cover your face (with the other hand, a book, etc.). Expected:
- After ~0.5s, overlay flips to **LOCKED** in red.
- Gestures stop firing.

- [ ] **Step 5: Verify lockout with a different person**

If a second person is available, have them stand in frame instead of
you. Expected: overlay shows **LOCKED**, their gestures are ignored.
If their face does match (false positive), adjust `MATCH_THRESHOLD`
upward in `handvol/face_profile.py` (try 0.95) and recommit; if your
own face is being incorrectly rejected (false negative), adjust
downward (try 0.88).

- [ ] **Step 6: Re-test regressions**

Run the full unit test suite one more time:
```bash
python -m pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 7: Commit any threshold tuning**

If `MATCH_THRESHOLD` was changed during Step 5:

```bash
git add handvol/face_profile.py
git commit -m "tune(face): adjust MATCH_THRESHOLD after manual calibration testing"
```

---

## Done Criteria

- All unit tests pass (`python -m pytest tests/ -q`).
- Manual checklist (Task 11) passes end-to-end.
- `data/face_profile.npz` is gitignored and not tracked.
- README documents the calibration step.
- Branch is ready to merge to `main`.
