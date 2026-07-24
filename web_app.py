# -*- coding: utf-8 -*-
"""
web_app.py -- Real-Time Web Dashboard for Drowsiness Detector
==============================================================
Architecture:
  - Background thread runs MediaPipe FaceMesh detection continuously
  - /video_feed   : MJPEG stream of processed frames (works in any browser <img>)
  - /stats_stream : Server-Sent Events pushing EAR, blinks, FPS, alerts ~4x/sec
  - /             : Serves the dashboard HTML

Usage:
    python web_app.py [--cam 0] [--ear 0.25] [--frames 20] [--port 5050]

Author  : [Your Name]
Course  : Computer Vision (BYOP)
"""

import cv2
import argparse
import time
import threading
import json
import os
import sys
from datetime import datetime

# ── Suppress TF import inside MediaPipe (protobuf version conflict workaround) ─
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# Stub out tensorflow so MediaPipe's optional TF import silently no-ops
import types as _types
_tf_stub = _types.ModuleType("tensorflow")
sys.modules.setdefault("tensorflow", _tf_stub)

import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, stream_with_context, request

from utils.ear import average_ear
from utils.visualizer import (
    draw_face_tracking,
    draw_status_bar,
    draw_alert_banner,
    draw_timestamp,
)

# ── MediaPipe landmark indices ───────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [ 33, 160, 158, 133, 153, 144]

app = Flask(__name__)

# ── Shared State (protected by a RLock) ─────────────────────────────────────
class DetectorState:
    """Thread-safe container for all detection metrics."""

    def __init__(self, ear_threshold: float = 0.20):
        self._lock              = threading.RLock()
        self.ear                = 0.30
        self.blink_count        = 0
        self.alert_active       = False
        self.warning_active     = False         # intermediate warning state
        self.ear_counter        = 0
        self.fps                = 0.0
        self.session_start      = datetime.now()
        self.alert_log          = []          # [{"time", "event", "ear"}, ...]
        self.ear_history        = []          # rolling EAR samples (max 300)
        self.latest_frame       = None        # raw JPEG bytes
        self.running            = False
        self.face_found         = False
        # ── Adaptive threshold ────────────────────────────────────
        self.ear_threshold      = ear_threshold  # live-adjustable
        self.is_calibrating     = False
        self.calibration_samples = []           # EAR samples during calib
        self.calib_countdown    = 0             # seconds remaining

    # Context manager for safe access
    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *_):
        self._lock.release()


state = DetectorState()


# ── Background Detection Thread ──────────────────────────────────────────────
def detection_loop(cam_idx: int, frames_threshold: int):
    """
    Runs MediaPipe FaceMesh + EAR logic in a daemon thread.
    Writes processed JPEG frames and stats into `state`.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_idx}.")
        with state:
            state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # Local counters (written to state each frame)
    ear_counter    = 0
    blink_counter  = 0
    alert_active   = False
    warning_active = False
    in_blink       = False
    tick           = 0

    fps_time  = time.time()
    fps_frame = 0
    fps       = 0.0

    with state:
        state.running = True

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear        = 0.30
        face_found = False

        if results.multi_face_landmarks:
            face_found = True
            landmarks  = results.multi_face_landmarks[0].landmark

            def lm_to_px(idx):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h)

            left_eye  = [lm_to_px(i) for i in LEFT_EYE_IDX]
            right_eye = [lm_to_px(i) for i in RIGHT_EYE_IDX]
            ear = average_ear(left_eye, right_eye)

            # Read current threshold from shared state (live-adjustable)
            with state:
                ear_thresh = state.ear_threshold
                # Collect calibration samples when calibrating
                if state.is_calibrating and ear > 0.10:
                    state.calibration_samples.append(ear)

            # Draw face tracking oval — colour reflects 3-stage alert level
            draw_face_tracking(frame, landmarks, w, h, alert_active, warning_active)

            # Blink detection
            if ear < ear_thresh:
                ear_counter += 1
                if not in_blink:
                    in_blink = True
            else:
                if in_blink:
                    blink_counter += 1
                    in_blink = False
                ear_counter = 0

            # ── 3-stage alert logic ──────────────────────────────────
            # warn_thresh = 75% of frames_threshold so yellow only appears
            # in the final ~500ms before drowsy fires, not on casual squints.
            warn_thresh = int(frames_threshold * 0.75)   # e.g. 15 of 20 frames

            # DROWSY (full alert)
            if ear_counter >= frames_threshold:
                warning_active = False
                if not alert_active:
                    alert_active = True
                    entry = {
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "event": "DROWSY",
                        "ear":   round(ear, 4),
                    }
                    with state:
                        state.alert_log.insert(0, entry)
                        if len(state.alert_log) > 50:
                            state.alert_log.pop()

            # WARNING (75 % of the way to drowsy)
            elif ear_counter >= warn_thresh:
                if not warning_active:          # edge: first frame of warning
                    warning_active = True
                    entry = {
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "event": "CAUTION",
                        "ear":   round(ear, 4),
                    }
                    with state:
                        state.alert_log.insert(0, entry)
                        if len(state.alert_log) > 50:
                            state.alert_log.pop()
                if alert_active:                # shouldn't normally happen, guard anyway
                    alert_active = False
                    entry = {
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "event": "RECOVERED",
                        "ear":   round(ear, 4),
                    }
                    with state:
                        state.alert_log.insert(0, entry)
                        if len(state.alert_log) > 50:
                            state.alert_log.pop()

            # AWAKE (clear)
            else:
                if alert_active:
                    # Drowsy → Awake
                    alert_active   = False
                    warning_active = False
                    entry = {
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "event": "RECOVERED",
                        "ear":   round(ear, 4),
                    }
                    with state:
                        state.alert_log.insert(0, entry)
                        if len(state.alert_log) > 50:
                            state.alert_log.pop()
                elif warning_active:
                    # Caution → Awake (eyes reopened before reaching full alert)
                    warning_active = False
                    entry = {
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "event": "RECOVERED",
                        "ear":   round(ear, 4),
                    }
                    with state:
                        state.alert_log.insert(0, entry)
                        if len(state.alert_log) > 50:
                            state.alert_log.pop()
        else:
            # Face lost — clear all alert states
            ear_counter    = 0
            warning_active = False
            if alert_active:
                alert_active = False

        # FPS calculation
        fps_frame += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps       = fps_frame / (now - fps_time)
            fps_frame = 0
            fps_time  = now

        # Draw overlays onto frame
        draw_status_bar(frame, ear, alert_active, warning_active, face_found)
        draw_alert_banner(frame, alert_active, tick)
        draw_timestamp(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        tick += 1

        # Push stats into shared state
        with state:
            state.ear            = round(ear, 4)
            state.blink_count    = blink_counter
            state.alert_active   = alert_active
            state.warning_active = warning_active
            state.ear_counter    = ear_counter
            state.fps            = round(fps, 1)
            state.face_found     = face_found
            state.ear_history.append(round(ear, 4))
            if len(state.ear_history) > 300:
                state.ear_history.pop(0)

        # Encode and push JPEG frame
        _, jpeg = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, 72]
        )
        with state:
            state.latest_frame = jpeg.tobytes()

    cap.release()
    face_mesh.close()
    print("[OK] Detection thread stopped.")


# ── Flask Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


def _mjpeg_generator():
    """Yields MJPEG frames from the shared state."""
    while True:
        with state:
            frame = state.latest_frame
        if frame is None:
            time.sleep(0.033)
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )
        time.sleep(0.033)   # cap at ~30 fps


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint — consumed by <img src="/video_feed">."""
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats_stream')
def stats_stream():
    """Server-Sent Events stream -- pushes JSON stats ~4x per second."""
    def _sse_generator():
        while True:
            with state:
                elapsed       = (datetime.now() - state.session_start)
                total_secs    = int(elapsed.total_seconds())
                h, rem        = divmod(total_secs, 3600)
                m, s          = divmod(rem, 60)
                payload = {
                    "ear":             state.ear,
                    "blinks":          state.blink_count,
                    "alert":           state.alert_active,
                    "warning":         state.warning_active,
                    "fps":             state.fps,
                    "face_found":      state.face_found,
                    "session":         f"{h:02d}:{m:02d}:{s:02d}",
                    "ear_history":     state.ear_history[-80:],
                    "alert_log":       state.alert_log[:15],
                    "ear_threshold":   state.ear_threshold,
                    "is_calibrating":  state.is_calibrating,
                    "calib_countdown": state.calib_countdown,
                }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.25)

    return Response(
        stream_with_context(_sse_generator()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Starts a 3-second calibration window.
    User should keep eyes naturally open.
    Threshold is set to 72% of the measured average EAR.
    """
    CALIB_SECONDS = 3
    CALIB_RATIO   = 0.72   # threshold = baseline * 0.72

    with state:
        if state.is_calibrating:
            return json.dumps({"status": "already_calibrating"}), 200
        state.is_calibrating      = True
        state.calibration_samples = []
        state.calib_countdown     = CALIB_SECONDS

    def _do_calibration():
        for remaining in range(CALIB_SECONDS, 0, -1):
            with state:
                state.calib_countdown = remaining
            time.sleep(1)

        with state:
            samples = list(state.calibration_samples)
            state.is_calibrating  = False
            state.calib_countdown = 0
            if len(samples) >= 5:   # need at least 5 samples
                baseline = sum(samples) / len(samples)
                new_thresh = round(baseline * CALIB_RATIO, 4)
                state.ear_threshold = new_thresh
                print(f"[Calibration] Baseline EAR: {baseline:.4f}  "
                      f"-> New threshold: {new_thresh:.4f}  "
                      f"(from {len(samples)} samples)")
            else:
                print("[Calibration] Not enough samples -- keep face visible next time.")

    threading.Thread(target=_do_calibration, daemon=True).start()
    return json.dumps({"status": "calibrating", "duration": CALIB_SECONDS}), 200


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Manually set EAR threshold from the dashboard slider."""
    data = json.loads(request.data)
    val  = float(data.get('threshold', 0.20))
    val  = max(0.10, min(0.40, val))   # clamp to safe range
    with state:
        state.ear_threshold = round(val, 4)
    return json.dumps({"status": "ok", "threshold": state.ear_threshold}), 200


# ── Entry Point ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Drowsiness Detector -- Web Dashboard"
    )
    p.add_argument("--cam",    type=int,   default=0,    help="Webcam index")
    p.add_argument("--ear",    type=float, default=0.20,
                   help="Initial EAR threshold (default: 0.20). Use 'Calibrate' button to personalise.")
    p.add_argument("--frames", type=int,   default=20,   help="Alert frame count")
    p.add_argument("--port",   type=int,   default=5050, help="Web server port")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    # Initialise state with the chosen threshold
    state.ear_threshold = args.ear

    # Start detection in background daemon thread
    detect_thread = threading.Thread(
        target=detection_loop,
        args=(args.cam, args.frames),
        daemon=True,
    )
    detect_thread.start()

    url = f"http://localhost:{args.port}"
    print("\n" + "=" * 52)
    print("  Drowsiness Detector -- Web Dashboard")
    print("=" * 52)
    print(f"  Dashboard : {url}")
    print(f"  Camera    : {args.cam}")
    print(f"  EAR Thresh: {args.ear} (adjustable via dashboard)")
    print(f"  Alert after {args.frames} consecutive frames")
    print("-" * 52)
    print("  Open the URL above in your browser.")
    print("  Tip: Use the Calibrate button for your eyes!")
    print("  Press Ctrl+C to stop.")
    print("=" * 52 + "\n")

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
