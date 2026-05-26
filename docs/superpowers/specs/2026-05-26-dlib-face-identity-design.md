# dlib Face Identity Swap — Design

**Date:** 2026-05-26
**Branch:** `face-recognition`
**Status:** Draft
**Supersedes (identity portion):** `2026-05-26-face-aware-hand-recognition-design.md`

## Problem

The current face-aware gate produces a "face embedding" by L2-normalizing
the 478 3D landmark positions from MediaPipe's Face Landmarker. In
practice:

- **Discrimination is poor.** After translation+scale normalization,
  landmark layouts of different humans look nearly identical. Same-face
  and different-face cosine similarities both sit above 0.97, leaving
  almost no margin for thresholding.
- **Facial expressions barely shift the score.** Smiles and frowns move
  landmarks a little, but the overall geometric layout is preserved, so
  the embedding is nearly constant — but for the wrong reason. It is
  also nearly constant *across people*.

The result: the gate effectively detects "a face is present" rather than
"my face is present," which does not meet the security requirement of
the original spec.

## Goal

Replace the identity *producer* with a real face-recognition model
(`face_recognition` / dlib ResNet-34, 128-D embeddings) so that
same-person similarity sits well above cross-person similarity, making
the gate genuinely identity-aware.

## Non-Goals

- GPU acceleration. dlib is installed CPU-only via a pre-built wheel.
  The design assumes CPU inference.
- Multi-profile support. Still single-user.
- Migrating existing v1 profile files. Re-calibration is required.
- Anti-spoofing / liveness detection. Out of scope for v2 as it was
  for v1.
- Replacing MediaPipe entirely. MediaPipe's Face Landmarker keeps a
  role as a fast per-frame face *detector* and overlay producer.

## Architecture

### Producer split

| Component | Role after change | Cadence |
|---|---|---|
| MediaPipe Gesture Recognizer | Hand gesture recognition (unchanged) | Every frame, async |
| MediaPipe Face Landmarker | Face *detection* (bbox), dots overlay, calibration UI feedback | Every frame, async |
| dlib `face_recognition` | Identity embedding (NEW) | ~3 Hz, async, coalescing |

Identity matching is no longer derived from MediaPipe landmarks. The
landmark vector → embedding helper is removed.

### Files

**New:**
- `handvol/face_identity.py` — wraps `face_recognition`. Two units:
  - `compute_identity_embedding(rgb_frame, bbox) -> np.ndarray | None`
    — synchronous; calls `face_recognition.face_encodings(...,
    known_face_locations=[bbox], num_jitters=1)`. Returns the first
    encoding L2-normalized (so cosine similarity == dot product), or
    `None` if dlib finds no face in the crop.
  - `IdentityEncoder` — async worker class. Owns its own thread.
    `submit(rgb, bbox)` is fire-and-forget, coalescing rate-limiter
    (see "Throttling" below). `latest() -> (embedding, ts_ns)` returns
    the most recent successful result, never blocks.
- `tests/test_face_identity.py` — tests for `IdentityEncoder`
  rate-limiter behavior using a mocked encoding function (no dlib
  invocation). Five tests; see Testing.

**Modified:**
- `handvol/face_detect.py` — keep `FaceEmbedder` (the MediaPipe
  wrapper) as-is for the bbox + dots role. **Remove**
  `landmarks_to_embedding` and `EXPECTED_LANDMARK_COUNT`. **Add**
  `landmarks_to_bbox(face_landmarks, frame_shape) -> tuple` that
  converts MediaPipe landmarks to `(top, right, bottom, left)` pixel
  coordinates suitable for `face_recognition`'s `known_face_locations`
  parameter.
- `handvol/face_profile.py` — bump `PROFILE_VERSION` from `1` to `2`.
  In `load()`, if the file's stored version is not equal to
  `PROFILE_VERSION`, log a warning and return `None`. Update
  `MATCH_THRESHOLD` from `0.92` to `0.92` for dlib (numerically the
  same but semantically a fresh starting point; tune empirically). All
  other API unchanged — the class is dimension-agnostic.
- `handvol/capture.py` — `GestureSource` gains an `IdentityEncoder`
  alongside the existing `FaceEmbedder`. On each frame:
  - submit to gesture recognizer (existing)
  - submit to MediaPipe Face Landmarker (existing)
  - if MediaPipe has produced face landmarks, pick the **largest** face
    (by bbox area) and submit `(rgb, bbox)` for that one face only to
    `IdentityEncoder`. Encoding all detected faces would multiply
    dlib's CPU cost and defeat the throttling budget; the v1
    "any-face-matches" rule is replaced with "the most prominent face
    must match" — which is also the safer default (a bystander cannot
    unlock by stepping in front of the user, only by displacing them
    visually).
  - assemble result tuple
  The result tuple becomes
  `(gesture, score, hand_landmarks, face_landmarks_list, identity_embedding)`
  where `identity_embedding` is a single 128-D vector (or `None`).
  `face_landmarks_list` keeps its current role for the dots overlay.
- `handvol/calibration.py` — replace the per-pose `embedder.latest()`
  call with a synchronous `compute_identity_embedding(rgb, bbox)` call
  on the most recent frame with a MediaPipe-detected face. Calibration
  itself does **not** throttle — the user is paused for the 2-second
  countdown anyway, and we want every pose's capture to use a fresh
  dlib run on the exact frame the user posed for.
- `handvol.pyw` — update result-tuple unpacking from
  `(gesture, score, landmarks, face_embs, face_lms)` to
  `(gesture, score, landmarks, face_lms, identity_emb)`. Match call
  becomes `profile.matches(identity_emb)` when `identity_emb is not
  None`. Add a startup check that `face_recognition` imports
  successfully; surface `SystemExit` with install instructions if not.

**Deleted:**
- `tests/test_face_embedding.py` — tests a function that no longer
  exists.

### Data flow (per frame)

```
camera frame ──► MediaPipe Gesture (async)        ──► gesture slot
              ├► MediaPipe FaceLandmarker (async) ──► face_landmarks
              │                                       + bbox slot
              └► IdentityEncoder.submit(rgb, bbox)
                   │
                   │ rate-limiter:
                   │   - drop if previous job still running
                   │   - drop if < MIN_INTERVAL_MS since last submit
                   ▼
                 worker thread (background):
                   face_recognition.face_encodings(
                       rgb,
                       known_face_locations=[bbox],
                       num_jitters=1)
                   ──► L2-normalize ──► 128-D embedding ──► identity slot

main loop reads:
   gesture, score, hand_landmarks, face_lms, identity_emb
   ↓
   if identity_emb is None: keep last_recognized (grace period)
   else: recognized = profile.matches(identity_emb)[0]
   ↓
   gate gesture dispatch, draw overlays (including dots from face_lms)
```

## Throttling

`IdentityEncoder` is a **coalescing** rate-limiter (never queues):

- `submit(rgb, bbox)` returns immediately.
- Drops the submission if:
  - the previous job is still running, OR
  - wall-clock time since the last *submitted* job is `< MIN_INTERVAL_MS`.
- One in-flight job at a time. The freshest call wins.
- `latest()` returns the most recent successful embedding plus
  `time.monotonic_ns()` of when it was produced.

**Constants** (in `face_identity.py`):
- `MIN_INTERVAL_MS = 333` — ~3 Hz cap on dlib calls during normal
  runtime.
- `JITTERS = 1` — `face_recognition`'s default; higher values average
  multiple slightly-perturbed crops for a more stable embedding at the
  cost of linear latency.

## Thresholding

`face_recognition`'s ResNet-34 produces 128-D embeddings with strong
identity separation. Typical numbers (cosine similarity on
L2-normalized vectors):

| Scenario | Typical cosine similarity |
|---|---|
| Same person, neutral vs neutral | 0.92-0.97 |
| Same person, neutral vs expression | 0.88-0.95 |
| Same person, varied lighting | 0.85-0.93 |
| Different person | 0.40-0.75 |

Starting `MATCH_THRESHOLD = 0.92`. Manual tuning during Task 11
verification likely lands between 0.88 and 0.94. The threshold value
is unchanged numerically from v1 but **does not have the same
meaning** — the discriminative gap is much wider with dlib, so the
same number sits in a very different part of the score distribution.

## Profile file format

- Path: `data/face_profile.npz` (unchanged).
- Contents: `embeddings` (shape `(N, 128)` float32, L2-normalized),
  `created_at` (string), `version` (int, now `2`).
- Old `(N, 1434)` files are not migrated. `FaceProfile.load()` rejects
  any file with `version != 2`. User must re-run calibration.

## Error handling & edge cases

| Scenario | Behavior |
|---|---|
| `face_recognition`/`dlib` not importable at app startup | `SystemExit` with install instructions, same pattern as the existing missing-model checks in `handvol.pyw`. |
| MediaPipe detects no face this frame | `IdentityEncoder.submit` is not called. `latest()` returns the previous embedding. Existing `NO_FACE_GRACE_FRAMES = 15` debouncer in `handvol.pyw` handles the lockout transition after ~500ms. |
| MediaPipe detects a face but dlib finds none in the crop | dlib returns an empty list. Worker logs once (rate-limited via `_warned` flag); leaves the latest embedding unchanged. Same downstream treatment as "no face." |
| Multiple faces in frame | Only the **largest** (by MediaPipe bbox area) is sent to dlib. If that face does not match the calibrated user, the gate stays locked even if a smaller face in the background would have matched. |
| Profile file is v1 | `FaceProfile.load()` returns `None`, logs `"face profile v1 is incompatible with v2 — please re-calibrate"`. Overlay shows `NO PROFILE`. |
| Profile file has unexpected embedding dim | `load()` returns `None` (existing shape check still applies). |
| Calibration pose fails dlib | The pose is retried — same path as the existing "no MediaPipe face detected" retry. Countdown restarts after `PER_POSE_TIMEOUT_SECONDS`. |
| `IdentityEncoder` worker thread hits an exception | Caught inside the worker; logged once; thread does not die. `latest()` keeps returning the last known good embedding. Eventually stale → grace-period lockout. |
| User submits frames faster than dlib can keep up | Coalescing drops submissions. No queue buildup, no rising latency. |

## Performance budget

| Step | Cost | Frequency |
|---|---|---|
| MediaPipe Gesture | ~5-15 ms | every frame |
| MediaPipe Face Landmarker | ~5-15 ms | every frame |
| dlib `face_encodings` with `known_face_locations` | ~30-80 ms | ~3 Hz (worker thread) |
| Cosine sim over (N=20, D=128) profile | sub-ms | per identity refresh |
| Lock-state staleness (worst case) | MIN_INTERVAL_MS + dlib latency ≈ 400-500 ms | n/a |

Main-loop FPS is unchanged (~30) because dlib runs on its own thread.
The added lock-state latency (~400-500 ms) is the same order as the
existing `NO_FACE_GRACE_FRAMES = 15` debouncer (~500 ms), so the
overall feel of lock/unlock transitions is consistent.

## Testing

### Unit tests

`tests/test_face_profile.py` (modify — add one test):
- `test_load_rejects_v1_profile` — write an `.npz` with `version=1`
  and arbitrary embeddings; assert `FaceProfile.load()` returns
  `None`.

`tests/test_face_identity.py` (new, no dlib required — uses a mocked
encoding function):
- `test_submit_drops_when_in_flight` — submit while a slow fake job is
  still running; the second submission is a no-op.
- `test_submit_drops_when_within_min_interval` — two submits within
  333 ms; the second is dropped.
- `test_submit_accepted_after_interval` — submit, wait > 333 ms,
  submit again; both run, `latest()` reflects the second.
- `test_latest_returns_most_recent_embedding` — after a fake job
  completes, `latest()` returns its produced embedding.
- `test_worker_exception_does_not_kill_thread` — fake function raises;
  next submission still runs.

`tests/test_face_embedding.py` (delete) — tests a removed function.

### Integration tests

None automated. dlib model load + a real face image is too heavy and
brittle for CI.

### Manual verification

1. With an old v1 profile on disk, app starts, overlay shows `NO
   PROFILE`, log warns about v1 incompatibility.
2. Run tray-menu calibration → 20 poses → profile saved at v2.
3. Similarity readout (`0.xx / 0.92` line under lock state) should now:
   - Sit at **0.93-0.97** for you, neutral.
   - Drop noticeably (**0.4-0.7**) for a different person in frame.
   - Move with expression changes but stay above threshold for normal
     smiling/talking.
4. Preview FPS stays ~30 — no visible scrub stutter.
5. Cover your face → after ~500 ms, overlay flips to `LOCKED`.
6. Tune `MATCH_THRESHOLD` if needed (likely 0.88-0.94).

## Out of scope (future work, unchanged from v1)

- Multi-profile support.
- Hand-skin-tone or hand-vs-face-size sanity checks.
- Anti-spoofing (liveness detection).
- Adaptive embedding updates ("learns over time").
- GPU acceleration of dlib.
