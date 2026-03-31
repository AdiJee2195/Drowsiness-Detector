"""
visualizer.py — OpenCV Drawing Utilities
=========================================
All HUD overlays, landmark renderings, and alert banners for
the drowsiness detector live here.
"""

import cv2
import numpy as np

# ─── Color Palette (BGR) ────────────────────────────────────────────────────
GREEN      = (80, 220, 100)
RED        = (50,  50, 230)
YELLOW     = (0,  210, 255)
CYAN       = (220, 200,  0)
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
DARK_PANEL = ( 15,  15,  25)
ORANGE     = ( 0,  140, 255)


# ─── HUD Panel ──────────────────────────────────────────────────────────────
def draw_overlay_panel(frame: np.ndarray, ear: float, blink_count: int,
                       alert_active: bool, fps: float,
                       alert_frame_count: int, alert_threshold: int) -> np.ndarray:
    """
    Draw a semi-transparent HUD panel in the top-left corner.

    Shows:  EAR value + bar, blink counter, FPS, status text,
            and a countdown bar to alert trigger.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 290, 210

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), DARK_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Border line
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), GREEN, 1)

    # ── Title ────────────────────────────────────────────────────────────────
    cv2.putText(frame, "DROWSINESS MONITOR", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
    cv2.line(frame, (10, 32), (280, 32), GREEN, 1)

    # ── EAR Value ────────────────────────────────────────────────────────────
    ear_color = RED if ear < 0.25 else YELLOW if ear < 0.30 else GREEN
    cv2.putText(frame, f"EAR    :  {ear:.3f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, ear_color, 2, cv2.LINE_AA)

    # EAR progress bar
    bar_bg   = (50, 50, 65)
    bar_full = 220
    bar_val  = max(0, min(int(ear / 0.42 * bar_full), bar_full))
    cv2.rectangle(frame, (10, 66),  (10 + bar_full, 80), bar_bg, -1)
    cv2.rectangle(frame, (10, 66),  (10 + bar_val,  80), ear_color, -1)
    # Threshold marker
    thresh_x = int(0.25 / 0.42 * bar_full) + 10
    cv2.line(frame, (thresh_x, 62), (thresh_x, 84), RED, 2)

    # ── Blink Count ──────────────────────────────────────────────────────────
    cv2.putText(frame, f"Blinks :  {blink_count}", (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 1, cv2.LINE_AA)

    # ── FPS ──────────────────────────────────────────────────────────────────
    fps_color = GREEN if fps > 20 else YELLOW if fps > 12 else RED
    cv2.putText(frame, f"FPS    :  {fps:.1f}", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, fps_color, 1, cv2.LINE_AA)

    # ── Alert Charge Bar ─────────────────────────────────────────────────────
    charge_label = "CHARGE :"
    charge_full  = 220
    charge_val   = max(0, min(int(alert_frame_count / alert_threshold * charge_full),
                              charge_full))
    charge_color = RED if alert_active else ORANGE if charge_val > charge_full // 2 else YELLOW
    cv2.putText(frame, charge_label, (10, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (10, 168), (10 + charge_full, 180), (50, 50, 65), -1)
    cv2.rectangle(frame, (10, 168), (10 + charge_val,  180), charge_color, -1)

    # ── Status ───────────────────────────────────────────────────────────────
    status_text  = "!!! DROWSY !!!" if alert_active else "ALERT"
    status_color = RED if alert_active else GREEN
    cv2.putText(frame, f"Status :  {status_text}", (10, 202),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, cv2.LINE_AA)

    return frame


# ─── Alert Banner ───────────────────────────────────────────────────────────
def draw_alert_banner(frame: np.ndarray, alert_active: bool,
                      tick: int) -> np.ndarray:
    """
    Draws a flashing red banner at the bottom of the frame when drowsy.
    The banner flashes at ~2 Hz using the tick counter.
    """
    if not alert_active:
        return frame

    h, w = frame.shape[:2]
    # Flash every 15 ticks
    if (tick // 15) % 2 == 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 75), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        text = "  !!  DROWSINESS DETECTED  —  PLEASE TAKE A BREAK  !!  "
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.68, 2)[0]
        text_x = max((w - text_size[0]) // 2, 5)
        cv2.putText(frame, text, (text_x, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, WHITE, 2, cv2.LINE_AA)
    return frame


# ─── Eye Landmarks ──────────────────────────────────────────────────────────
def draw_eye_landmarks(frame: np.ndarray, eye_points: list,
                       color: tuple) -> np.ndarray:
    """
    Draw the convex hull of eye landmark points.
    Also draws a small dot at each landmark.

    Args:
        frame:      BGR image array
        eye_points: list of (x, y) tuples
        color:      BGR color tuple
    """
    pts = np.array([(int(p[0]), int(p[1])) for p in eye_points])
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, color, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, tuple(pt), 2, color, -1, cv2.LINE_AA)
    return frame


# ─── Timestamp & Watermark ──────────────────────────────────────────────────
def draw_timestamp(frame: np.ndarray, timestamp: str) -> np.ndarray:
    """Draws a timestamp in the bottom-right corner."""
    h, w = frame.shape[:2]
    cv2.putText(frame, timestamp, (w - 230, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return frame
