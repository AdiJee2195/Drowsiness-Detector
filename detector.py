"""
detector.py — Real-Time Driver Drowsiness Detection System
===========================================================
Uses MediaPipe FaceMesh (468 landmarks) to track eye landmarks
in real time from a webcam feed, computes the Eye Aspect Ratio (EAR),
and triggers an audio + visual alert when prolonged eye closure
(drowsiness) is detected.

Usage:
    python detector.py [--cam 0] [--ear 0.25] [--frames 20] [--no-sound]

Controls (while running):
    Q / ESC  — quit
    R        — reset blink counter
    S        — toggle sound on/off

Author  : [Your Name]
Course  : Computer Vision (BYOP)
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import csv
import os
import threading
from datetime import datetime

from utils.ear import average_ear
from utils.visualizer import (
    draw_overlay_panel,
    draw_alert_banner,
    draw_eye_landmarks,
    draw_timestamp,
)

# ─── MediaPipe Eye Landmark Indices (from 468-point FaceMesh) ───────────────
# Each list contains 6 indices ordered as:
#   [left_corner, top_left, top_right, right_corner, bot_right, bot_left]
# Chosen to match the EAR formula (Soukupová & Čech, 2016).

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [ 33, 160, 158, 133, 153, 144]


# ─── Alarm Thread ────────────────────────────────────────────────────────────
def _play_alarm(alarm_path: str, stop_event: threading.Event) -> None:
    """Background thread: loops alarm sound until stop_event is set."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play(-1)   # -1 = loop indefinitely
        stop_event.wait()
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception:
        # Fallback: Windows system beep (no dependencies)
        try:
            import winsound
            while not stop_event.is_set():
                winsound.Beep(880, 300)
                time.sleep(0.05)
                winsound.Beep(440, 300)
                time.sleep(0.05)
        except Exception:
            pass  # Silent fallback — visual alert still works


class AlarmController:
    """Manages the alarm sound lifecycle (start / stop, no double-triggers)."""

    def __init__(self, alarm_path: str):
        self.alarm_path  = alarm_path
        self._thread     = None
        self._stop_event = threading.Event()
        self.is_playing  = False

    def start(self):
        if not self.is_playing:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=_play_alarm,
                args=(self.alarm_path, self._stop_event),
                daemon=True,
            )
            self._thread.start()
            self.is_playing = True

    def stop(self):
        if self.is_playing:
            self._stop_event.set()
            self.is_playing = False


# ─── CSV Logger ──────────────────────────────────────────────────────────────
class DrowsinessLogger:
    """Appends drowsiness events to logs/drowsiness_log.csv."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "drowsiness_log.csv")
        # Write header if file is new
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "event", "ear_value"])

    def log(self, event: str, ear: float):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             event, f"{ear:.4f}"])


# ─── Argument Parser ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-Time Drowsiness Detection using MediaPipe + EAR"
    )
    p.add_argument("--cam",      type=int,   default=0,
                   help="Webcam index (default: 0)")
    p.add_argument("--ear",      type=float, default=0.25,
                   help="EAR threshold below which eyes are considered closed "
                        "(default: 0.25)")
    p.add_argument("--frames",   type=int,   default=20,
                   help="Consecutive frames below threshold before alert "
                        "(default: 20)")
    p.add_argument("--no-sound", action="store_true",
                   help="Disable audio alarm (visual only)")
    p.add_argument("--alarm",    type=str,   default="alarm.wav",
                   help="Path to alarm WAV file (default: alarm.wav)")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    logger = DrowsinessLogger()
    alarm  = AlarmController(args.alarm)
    sound_enabled = not args.no_sound

    # ── MediaPipe FaceMesh setup ─────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces       = 1,
        refine_landmarks    = True,
        min_detection_confidence = 0.6,
        min_tracking_confidence  = 0.6,
    )

    # ── Webcam setup ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.cam}.")
        print("        Try a different --cam value (e.g., --cam 1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # ── State variables ──────────────────────────────────────────────────────
    ear_counter   = 0     # consecutive frames with low EAR
    blink_counter = 0     # total blink count
    alert_active  = False
    in_blink      = False # True while eyes are currently closed
    tick          = 0     # frame counter (for flashing effects)

    # FPS tracking
    fps_time  = time.time()
    fps_frame = 0
    fps       = 0.0

    print("\n" + "=" * 55)
    print("  [CAR]  Drowsiness Detection System  --  RUNNING")
    print("=" * 55)
    print(f"  Camera     : {args.cam}")
    print(f"  EAR Thresh : {args.ear}")
    print(f"  Alert after: {args.frames} consecutive frames")
    print(f"  Sound      : {'ON (' + args.alarm + ')' if sound_enabled else 'OFF'}")
    print("-" * 55)
    print("  Controls:  Q / ESC = Quit  |  R = Reset blinks  |  S = Toggle sound")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame -- retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # mirror for natural feel
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Face landmark detection ──────────────────────────────────────────
        results = face_mesh.process(rgb)
        ear     = 0.30   # default (no face)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract eye coords in pixel space
            def lm_to_px(idx):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h)

            left_eye  = [lm_to_px(i) for i in LEFT_EYE_IDX]
            right_eye = [lm_to_px(i) for i in RIGHT_EYE_IDX]

            ear = average_ear(left_eye, right_eye)

            # Draw eye landmark outlines
            eye_color = (50, 50, 230) if ear < args.ear else (80, 220, 100)
            draw_eye_landmarks(frame, left_eye,  eye_color)
            draw_eye_landmarks(frame, right_eye, eye_color)

            # ── Blink detection (EAR rises back above threshold) ─────────────
            if ear < args.ear:
                ear_counter += 1
                if not in_blink:
                    in_blink = True
            else:
                if in_blink:
                    blink_counter += 1
                    in_blink = False
                ear_counter = 0

            # ── Drowsiness alert ─────────────────────────────────────────────
            if ear_counter >= args.frames:
                if not alert_active:
                    alert_active = True
                    logger.log("DROWSY_START", ear)
                    print(f"[!! ALERT] Drowsiness detected at "
                          f"{datetime.now().strftime('%H:%M:%S')}  EAR={ear:.3f}")
                    if sound_enabled:
                        alarm.start()
            else:
                if alert_active:
                    alert_active = False
                    logger.log("DROWSY_END", ear)
                    print(f"[OK]       Driver alert again at "
                          f"{datetime.now().strftime('%H:%M:%S')}")
                    alarm.stop()

        else:
            # No face detected
            ear_counter = 0
            if alert_active:
                alert_active = False
                alarm.stop()

        # ── FPS calculation ───────────────────────────────────────────────────
        fps_frame += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps       = fps_frame / (now - fps_time)
            fps_frame = 0
            fps_time  = now

        # ── HUD rendering ─────────────────────────────────────────────────────
        draw_overlay_panel(frame, ear, blink_counter, alert_active, fps,
                           ear_counter, args.frames)
        draw_alert_banner(frame, alert_active, tick)
        draw_timestamp(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        # "SOUND OFF" badge
        if not sound_enabled:
            cv2.putText(frame, "[MUTE] SOUND OFF", (w - 195, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1,
                        cv2.LINE_AA)

        tick += 1

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Drowsiness Detection System", frame)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):        # Q or ESC → quit
            break
        elif key == ord("r"):            # R → reset blinks
            blink_counter = 0
            ear_counter   = 0
            print("[INFO] Blink counter reset.")
        elif key == ord("s"):            # S → toggle sound
            sound_enabled = not sound_enabled
            if not sound_enabled:
                alarm.stop()
            print(f"[INFO] Sound {'enabled' if sound_enabled else 'disabled'}.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    alarm.stop()
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print(f"\n[OK] Session ended. Log saved -> {logger.path}")
    print(f"    Total blinks detected : {blink_counter}")


if __name__ == "__main__":
    main()
