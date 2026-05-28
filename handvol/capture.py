import math
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"

OK_SIGN_PINCH_THRESHOLD = 0.06

SIDE_THUMB_HORIZ_RATIO = 1.2   # |dx| must exceed |dy| by this factor
SIDE_THUMB_EXTEND_RATIO = 1.3  # tip-to-wrist must exceed cmc-to-wrist by this factor
SIDE_THUMB_CURL_RATIO = 1.2    # fingertip-to-wrist must stay within this * mcp-to-wrist

HANG_LOOSE_PINKY_EXTEND_RATIO = 1.5  # pinky tip-to-wrist must exceed mcp-to-wrist by this factor
HANG_LOOSE_CURL_RATIO = 1.2          # curled fingers' tip-to-wrist must stay within this * mcp-to-wrist


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


def _finger_states(landmarks):
    """Return (thumb_open, index_open, middle_open, ring_open, pinky_open).

    Non-thumb fingers count as open when the tip is above (smaller y) the PIP joint
    — assumes a roughly upright hand, which matches how users hold up numbers.
    Thumb uses a distance check against the index MCP so it works for either hand
    without caring about left/right orientation.
    """
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    # Thumb open: tip is meaningfully farther from the wrist than the MCP joint.
    # Anchoring on the wrist (instead of the index MCP) separates open vs tucked
    # for either hand and any thumb direction — same trick _detect_side_thumb uses.
    wrist = landmarks[0]
    tip_d = math.hypot(landmarks[4].x - wrist.x, landmarks[4].y - wrist.y)
    mcp_d = math.hypot(landmarks[2].x - wrist.x, landmarks[2].y - wrist.y)
    thumb = tip_d > mcp_d * 1.5
    return thumb, index, middle, ring, pinky


def _detect_three_fingers(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False
    t, i, m, r, p = _finger_states(landmarks)
    return i and m and r and not p and not t


def _detect_four_fingers(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False
    t, i, m, r, p = _finger_states(landmarks)
    return i and m and r and p and not t


def _detect_middle_finger(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False
    _, i, m, r, p = _finger_states(landmarks)
    return m and not i and not r and not p


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


def _detect_hang_loose(landmarks):
    """Return True for a shaka: thumb + pinky extended, index/middle/ring curled.

    Layered on top of _finger_states so direction/handedness don't matter, with
    extra ratio checks to demand a real extension on the pinky and tight curls
    on the middle three — same scale-invariant trick _detect_side_thumb uses.
    """
    if not landmarks or len(landmarks) < 21:
        return False
    t, i, m, r, p = _finger_states(landmarks)
    if not (t and p and not i and not m and not r):
        return False

    wrist = landmarks[0]
    pinky_tip_d = math.hypot(landmarks[20].x - wrist.x, landmarks[20].y - wrist.y)
    pinky_mcp_d = math.hypot(landmarks[17].x - wrist.x, landmarks[17].y - wrist.y)
    if pinky_tip_d < pinky_mcp_d * HANG_LOOSE_PINKY_EXTEND_RATIO:
        return False

    for mcp_idx, tip_idx in ((5, 8), (9, 12), (13, 16)):
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        mcp_to_wrist = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
        tip_to_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        if tip_to_wrist > mcp_to_wrist * HANG_LOOSE_CURL_RATIO:
            return False
    return True


FINGER_COUNT = {
    "Closed_Fist": 0,
    "Pointing_Up": 1,
    "Victory": 2,
    "three_fingers": 3,
    "four_fingers": 4,
    "Open_Palm": 5,
}


def _resolve_hand(landmarks, handedness, mp_name, mp_score):
    """Pick the best label for a single hand, preferring our custom detectors
    over MediaPipe's built-in category. Custom detectors run first so that, e.g.,
    a 4-fingers-up gesture lands as 'four_fingers' even if MediaPipe would have
    called it Open_Palm."""
    if _detect_ok_sign(landmarks):
        return "OK_sign", 1.0
    side = _detect_side_thumb(landmarks, handedness)
    if side is not None:
        return side, 1.0
    if _detect_middle_finger(landmarks):
        return "middle_finger", 1.0
    if _detect_hang_loose(landmarks):
        return "hang_loose", 1.0
    if _detect_three_fingers(landmarks):
        return "three_fingers", 1.0
    if _detect_four_fingers(landmarks):
        return "four_fingers", 1.0
    return mp_name, mp_score


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
        hands = []  # list of (name, score, landmarks)
        n = len(result.hand_landmarks) if result.hand_landmarks else 0
        for i in range(n):
            lm = result.hand_landmarks[i]
            hd = None
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                hd = result.handedness[i][0].category_name
            mp_name, mp_score = "None", 0.0
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                top = result.gestures[i][0]
                mp_name = top.category_name or "None"
                mp_score = top.score
            name, score = _resolve_hand(lm, hd, mp_name, mp_score)
            hands.append((name, score, lm))

        if len(hands) >= 2:
            n0, s0, lm0 = hands[0]
            n1, s1, lm1 = hands[1]
            # Two-hand combinations are only meaningful when both hands have a
            # recognized "count contribution"; otherwise we fall back to hand 0
            # so existing single-hand controls (OK_sign scrub, side-thumbs) keep
            # working even when a second hand wanders into frame.
            if n0 == "middle_finger" and n1 == "middle_finger":
                gesture_name, score, primary_lm = "double_middle_finger", 1.0, lm0
            else:
                c0 = FINGER_COUNT.get(n0)
                c1 = FINGER_COUNT.get(n1)
                if c0 is not None and c1 is not None and 1 <= c0 + c1 <= 10:
                    gesture_name, score, primary_lm = f"Number_{c0 + c1}", 1.0, lm0
                else:
                    # Prefer whichever hand is making a scrub-eligible gesture so
                    # OK_sign scrubbing still anchors on the right hand's tips.
                    if n1 == "OK_sign" and n0 != "OK_sign":
                        gesture_name, score, primary_lm = n1, s1, lm1
                    else:
                        gesture_name, score, primary_lm = n0, s0, lm0
        elif len(hands) == 1:
            gesture_name, score, primary_lm = hands[0]
        else:
            gesture_name, score, primary_lm = "None", 0.0, None

        all_lm = [h[2] for h in hands]
        with self._lock:
            self._latest = (gesture_name, score, primary_lm, all_lm)

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
            num_hands=2,
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
