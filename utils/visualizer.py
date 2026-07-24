"""
visualizer.py — OpenCV Drawing Utilities
=========================================
Clean, minimal overlays for the drowsiness detector.
  - Face oval tracking outline (replaces eye dots)
  - Minimal status bar at the bottom of the frame
  - Flashing alert banner when drowsy
"""

import cv2
import numpy as np

# ─── Color Palette (BGR) ────────────────────────────────────────────────────
GREEN  = ( 60, 210,  80)
RED    = ( 45,  55, 220)
YELLOW = (  0, 200, 240)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)

# MediaPipe face oval landmark indices (subset of the 468-point mesh)
# These trace the outer boundary of the face.
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

# A few key points to show as subtle tracking dots (nose, chin, cheeks)
KEY_POINTS_IDX = [1, 4, 152, 234, 454]


# ─── Face Tracking Overlay ──────────────────────────────────────────────────
def draw_face_tracking(frame: np.ndarray, landmarks, w: int, h: int,
                       alert_active: bool, warning_active: bool = False) -> np.ndarray:
    """
    Draw a clean face oval + key point dots.
    Colour: green (awake) → yellow (warning) → red (drowsy).
    """
    color = RED if alert_active else YELLOW if warning_active else GREEN

    # Convert face oval indices to pixel coords
    oval_pts = []
    for idx in FACE_OVAL_IDX:
        lm = landmarks[idx]
        oval_pts.append((int(lm.x * w), int(lm.y * h)))

    # Draw the oval outline
    for i in range(len(oval_pts) - 1):
        cv2.line(frame, oval_pts[i], oval_pts[i + 1], color, 1, cv2.LINE_AA)

    # Corner accent marks at top and chin for a "tracking target" feel
    top   = oval_pts[0]
    chin  = oval_pts[17] if len(oval_pts) > 17 else oval_pts[-1]
    left  = oval_pts[9]  if len(oval_pts) > 9  else oval_pts[0]
    right = oval_pts[27] if len(oval_pts) > 27 else oval_pts[-1]

    tick_len = 10
    tick_w   = 2
    for anchor, direction in [
        (top,   [(0, -tick_len), (-tick_len, 0), (tick_len, 0)]),
        (chin,  [(0,  tick_len), (-tick_len, 0), (tick_len, 0)]),
        (left,  [(-tick_len, 0), (0, -tick_len), (0, tick_len)]),
        (right, [( tick_len, 0), (0, -tick_len), (0, tick_len)]),
    ]:
        ax, ay = anchor
        for dx, dy in direction:
            cv2.line(frame, (ax, ay), (ax + dx, ay + dy), color, tick_w, cv2.LINE_AA)

    # Key point dots (nose tip, chin, cheeks)
    for idx in KEY_POINTS_IDX:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)

    return frame


# ─── Minimal Status Bar ─────────────────────────────────────────────────────
def draw_status_bar(frame: np.ndarray, ear: float, alert_active: bool,
                    warning_active: bool, face_found: bool) -> np.ndarray:
    """
    Thin status bar at the very bottom of the frame.
    Shows EAR value and 3-stage status.
    """
    h, w = frame.shape[:2]
    bar_h = 28

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (12, 14, 20), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if not face_found:
        cv2.putText(frame, "No face detected", (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, GRAY, 1, cv2.LINE_AA)
        return frame

    if alert_active:
        ear_color  = RED
        status_str = "DROWSY"
        status_col = RED
    elif warning_active:
        ear_color  = YELLOW
        status_str = "CAUTION"
        status_col = YELLOW
    else:
        ear_color  = GREEN if ear >= 0.28 else YELLOW
        status_str = "AWAKE"
        status_col = GREEN

    cv2.putText(frame, f"EAR {ear:.3f}", (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, ear_color, 1, cv2.LINE_AA)
    cv2.putText(frame, status_str, (w - 90, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 2, cv2.LINE_AA)

    return frame


# ─── Alert Banner ───────────────────────────────────────────────────────────
def draw_alert_banner(frame: np.ndarray, alert_active: bool,
                      tick: int) -> np.ndarray:
    """Flashing red banner at the top of the frame when drowsy."""
    if not alert_active:
        return frame

    h, w = frame.shape[:2]
    if (tick // 12) % 2 == 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 180), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        text = "DROWSINESS DETECTED — PLEASE TAKE A BREAK"
        sz   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(frame, text, ((w - sz[0]) // 2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
    return frame


# ─── Timestamp ──────────────────────────────────────────────────────────────
def draw_timestamp(frame: np.ndarray, timestamp: str) -> np.ndarray:
    """Subtle timestamp in the top-right corner."""
    h, w = frame.shape[:2]
    cv2.putText(frame, timestamp, (w - 228, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
    return frame
