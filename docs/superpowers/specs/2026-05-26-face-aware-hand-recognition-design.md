# Face-Aware Hand Recognition Filter — Design

**Date:** 2026-05-26
**Branch:** `face-recognition`
**Status:** Draft

## Problem

HandVol currently responds to *any* hand gesture in the camera frame. In a
shared room or office, anyone walking by with a visible hand can change
the user's system volume, mute/unmute, toggle playback, or focus Spotify.
This is both a UX annoyance and a minor security concern: gesture-based
control of the user's machine should require the user's own presence.

## Goal

Add a face-recognition gate so that hand gestures only trigger actions
when the **calibrated user's face** is recognized in the frame. If the
user's face is not recognized (or no face is detected), gestures are
ignored and the system enters a "locked" state.

## Non-Goals

- Multi-profile support (one calibrated user only, for v1).
- Robustness against deliberate spoofing (printed photos, deepfakes).
- Recognition of the user with substantially different appearance from
  calibration (e.g. heavy makeup, masks). Standard expression changes
  (smiles, frowns, talking) must work, but the calibration captures a
  neutral resting face only.
- Recognition of *which* hand belongs to the user when multiple hands
  are in frame. v1 simply uses the existing `num_hands=1` MediaPipe
  setting; the face gate is what enforces ownership.

## User Flow

### First-run calibration
1. User runs `python -m handvol.calibration` (or via a tray menu item).
2. A camera preview opens with on-screen prompts.
3. The user is guided through ~20 face captures at multiple angles and
   distances, all at a **neutral resting expression**:
   - Center (looking at camera) — 3 distances (close, medium, far)
   - Up, down, left, right
   - Diagonals: up-left, up-right, down-left, down-right
   - Profile-ish views: turn head to show more of the left cheek/ear,
     then more of the right cheek/ear
4. For each capture, MediaPipe Face Landmarker generates a face
   embedding. All embeddings are stored.
5. The calibration file is saved to a user config directory.
6. After calibration, HandVol runs normally; the face gate is active.

### Normal runtime
1. Each frame is processed by MediaPipe (existing flow).
2. Face Detection runs in parallel with the gesture recognizer.
3. If a face is detected:
   - Generate its embedding.
   - Compute the maximum cosine similarity against all stored
     calibration embeddings.
   - If `max_similarity >= MATCH_THRESHOLD`, the user is **recognized**;
     gestures are allowed.
   - Otherwise, gestures are blocked.
4. If no face is detected at all, gestures are blocked.
5. The tray icon overlay reflects the gate state (recognized / locked).

## Architecture

### New module: `handvol/face_profile.py`

```
FaceProfile
  fields:
    embeddings: np.ndarray  # shape (N, D), one row per calibration capture
  classmethods:
    load(path: Path) -> FaceProfile | None
    create_empty() -> FaceProfile
  methods:
    add_capture(embedding: np.ndarray) -> None
    save(path: Path) -> None
    matches(embedding: np.ndarray) -> tuple[bool, float]
        # returns (is_match, max_similarity)
```

- Storage location: `%APPDATA%/handvol/face_profile.npz` on Windows.
  Single `.npz` file holding the embeddings array plus metadata
  (creation date, capture count, calibration version).
- `matches()` computes cosine similarity vector and returns whether
  the max meets the threshold.

### New module: `handvol/face_detect.py`

Thin wrapper around MediaPipe's Face Landmarker (or Face Embedder if
available in the installed MediaPipe version):

```
FaceEmbedder
  __init__(model_path)
  embed(frame_rgb) -> np.ndarray | None
```

The embedder runs in the same `LIVE_STREAM` async style as the existing
gesture recognizer to avoid blocking the capture loop. Result is stashed
in a latest-result slot, identical to the existing `_on_result` pattern
in `capture.py`.

### Updated module: `handvol/capture.py`

- `GestureSource.open()` constructs a second MediaPipe task: a Face
  Landmarker / Embedder in `LIVE_STREAM` mode.
- `GestureSource.read()` submits each frame to **both** the gesture
  recognizer and the face embedder.
- The result tuple grows from `(gesture_name, score, landmarks)` to
  `(gesture_name, score, landmarks, face_recognized: bool)`.
- A new `_face_lock` and `_latest_face` slot mirror the existing
  `_lock` / `_latest` for the gesture stream.
- A loaded `FaceProfile` (or `None` if no profile exists) is passed
  into `GestureSource.__init__`.

### Updated entry point: `handvol.pyw`

- On startup, attempt to load the face profile.
- If no profile exists, the app starts in a "needs calibration"
  state: gestures are blocked, and the tray menu shows a "Calibrate
  face..." item that launches `handvol.calibration`.
- The main loop reads `face_recognized` from the capture result and
  gates the existing event dispatch on it.
- The overlay (`overlay.py`) gets a small lock-state indicator.

### New module: `handvol/calibration.py`

- Implements a standalone calibration flow (`python -m
  handvol.calibration`).
- Opens the camera, shows a step-by-step prompt ("Look up-and-right",
  etc.) with a countdown for each pose.
- Captures a face embedding per pose, asserts a face was detected,
  and retries if not.
- Writes the resulting `FaceProfile` to disk.
- Can be re-run to overwrite an existing profile (`--force`) or to
  add additional captures (`--append`).

## Data Flow

```
frame ──► gesture recognizer (async) ──► gesture result slot
   │
   └────► face embedder (async) ─────► face result slot
                                            │
                              latest face embedding
                                            │
                                FaceProfile.matches(emb)
                                            │
                                  face_recognized: bool
                                            │
main loop reads gesture + face_recognized   │
   │                                        ▼
   └── if face_recognized: dispatch gesture
       else: drop gesture, show locked overlay
```

## Configuration & Thresholds

- `MATCH_THRESHOLD`: cosine similarity threshold for "recognized".
  Start at `0.6`; tune empirically during testing.
- `NO_FACE_GRACE_FRAMES`: number of frames without a detected face
  before locking out gestures. Start at `15` (~0.5s at 30fps) to avoid
  flicker when the user briefly looks away or face detection drops a
  frame.
- All thresholds live as module constants at the top of
  `face_profile.py` / `face_detect.py`, mirroring the existing
  `OK_SIGN_PINCH_THRESHOLD` pattern in `capture.py`.

## Error Handling & Edge Cases

| Scenario | Behavior |
|---|---|
| No profile file on startup | Block all gestures; surface "Calibrate face" tray item |
| Profile file corrupted | Log warning, block gestures, treat as no profile |
| Face detection fails for N frames | After `NO_FACE_GRACE_FRAMES`, block gestures |
| Multiple faces in frame | Compute similarity for each; if **any** face matches, allow gestures |
| Low light / face too small | Face embedder returns `None`; treated as "no face" |
| Calibration interrupted | Don't write partial profile; require re-run |

## Performance Considerations

- MediaPipe face embedder model size and inference cost should be
  comparable to the existing gesture recognizer (~5-15ms per frame
  on a modern CPU).
- Both tasks run async via MediaPipe `LIVE_STREAM` callbacks; the main
  read loop stays non-blocking.
- Cosine similarity over ~20 embeddings of dim ~192 is a sub-millisecond
  numpy op — no caching needed.

## Testing

### Unit tests (no camera required)
- `tests/test_face_profile.py`
  - Round-trip save/load preserves embeddings.
  - `matches()` returns `True` for an embedding identical to a stored
    one (similarity = 1.0).
  - `matches()` returns `False` for an orthogonal embedding
    (similarity = 0.0).
  - `matches()` returns `False` for an empty/missing profile.

### Integration tests
- `tests/test_calibration.py`: feed fake frames through a stubbed
  embedder, assert the resulting profile has the right number of
  captures and is loadable.

### Manual tests
- Calibrate, then verify gestures work normally with only the user in
  frame.
- Have a second person stand in frame; their gestures should be
  ignored, the user's should still fire.
- Cover the user's face (e.g., hold up a hand); verify lockout
  triggers after the grace period.
- Re-test all existing gestures (point, fist, palm, victory, OK sign,
  thumbs up/down, ILoveYou) to confirm no regressions.

## Open Questions

- Which MediaPipe face task version is installed in the project's
  requirements? Need to verify `Face Landmarker` is available, or
  pick an alternative embedder (e.g., `face_recognition` / `dlib`,
  ONNX ArcFace) during implementation.
- Should the lockout state be visually distinct in the overlay
  (e.g., red border around the preview window) or just a small icon?
  Default to a small icon for v1; reconsider after manual testing.

## Out of Scope (Future Work)

- Multi-profile support with profile switching.
- Hand-skin-tone matching as a second-factor filter.
- Hand-size-vs-face-size sanity check to reject distant onlookers'
  hands even when the user's face is in frame.
- Anti-spoofing (liveness detection).
- Adaptive embedding updates ("learns over time").
