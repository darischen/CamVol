import cv2


WHITE = (255, 255, 255)
GREEN = (80, 220, 120)
YELLOW = (60, 220, 240)
CYAN = (255, 255, 0)
GRAY = (140, 140, 140)
RED = (80, 80, 240)


def _put(frame, text, org, color=WHITE, scale=0.6, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_state(frame, state):
    _put(frame, f"STATE: {state}", (10, 28), GREEN, 0.7)


def draw_gesture(frame, name, score=None):
    label = f"GESTURE: {name}"
    if score is not None and name != "None":
        label += f" ({score:.2f})"
    _put(frame, label, (10, 56), YELLOW, 0.6)


def draw_volume(frame, vol):
    h, w = frame.shape[:2]
    text = f"VOL: {int(round(vol))}" if vol is not None else "VOL: --"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    _put(frame, text, (w - tw - 12, 28), WHITE, 0.7)


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    text = f"{fps:5.1f} fps"
    _put(frame, text, (w - 110, h - 12), GRAY, 0.5, 1)


def draw_landmarks(frame, landmarks, color=GRAY):
    """Draw the 21 hand landmarks faintly. landmarks is list of NormalizedLandmark."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    # Simple skeleton: thumb, index, middle, ring, pinky chains + palm
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [5, 9, 10, 11, 12],
        [9, 13, 14, 15, 16],
        [13, 17, 18, 19, 20],
        [0, 17],
    ]
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_hold_timer(frame, action, elapsed_seconds, threshold=5.0):
    """Draw hold timer for destructive gestures (e.g., 'RESTART 2.3s / 5.0s')."""
    label = f"{action} {elapsed_seconds:.1f}s / {threshold:.1f}s"
    _put(frame, label, (10, 84), RED, 0.6)


def draw_scrub_indicator(frame, anchor_y_norm, tip_y_norm, tip_x_norm):
    """Cyan horizontal line at anchor, dot at tip, vertical connector."""
    if anchor_y_norm is None or tip_y_norm is None:
        return
    h, w = frame.shape[:2]
    ay = int(anchor_y_norm * h)
    ty = int(tip_y_norm * h)
    tx = int(tip_x_norm * w) if tip_x_norm is not None else w // 2
    cv2.line(frame, (0, ay), (w, ay), CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, (tx, ay), (tx, ty), CYAN, 2, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 7, CYAN, 2, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 2, CYAN, -1, cv2.LINE_AA)
