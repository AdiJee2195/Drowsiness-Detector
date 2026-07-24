# 🚗 Real-Time Driver Drowsiness Detection System

> A Computer Vision system that monitors a driver's eyes through a webcam, detects drowsiness in real-time using facial landmark geometry, and delivers progressive visual and audio alerts — all through a live web dashboard accessible from any browser on the local network.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)](https://mediapipe.dev)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Alert Stages](#-alert-stages)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Dashboard](#-dashboard)
- [Configuration](#-configuration)
- [Technical Details](#-technical-details)
- [Dependencies](#-dependencies)

---

## 🎯 Problem Statement

Drowsy driving is a leading cause of road accidents worldwide. According to NHAI data, fatigue-related crashes account for a significant proportion of highway fatalities in India. Drivers often fall asleep without realizing it — making **early detection critical**.

This system provides a real-time, non-intrusive computer vision solution: it watches the driver's eyes through a webcam, progressively warns them as eyes begin to close, and fires a full alert if sustained eye closure is detected — giving the driver time to pull over safely.

---

## 🧠 How It Works

The system uses a four-stage pipeline:

```
Webcam Frame → Face Landmark Detection → EAR Calculation → 3-Stage Alert Logic → Web Dashboard
```

### 1. Face Landmark Detection
**MediaPipe FaceMesh** detects 468 facial landmarks per frame. Six specific landmarks per eye are extracted to calculate the EAR. The full face oval outline (36 landmarks) is drawn as a tracking indicator.

### 2. Eye Aspect Ratio (EAR)
The EAR quantifies how open an eye is using a geometric ratio:

```
        ||p2 - p6|| + ||p3 - p5||
EAR =  ───────────────────────────
              2 × ||p1 - p4||
```

Where `p1–p6` are the six eye landmark coordinates (horizontal and vertical extents).

| State         | EAR Value |
|---------------|-----------|
| Eyes open     | ~0.30     |
| Eyes blinking | ~0.15     |
| Eyes closed   | ~0.00     |

📌 Reference: *Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", CVWW 2016*

### 3. Three-Stage Alert Logic

Rather than a binary awake/drowsy switch, the system escalates through three stages as eye-closed frames accumulate:

| Frames below threshold | Stage | Colour |
|---|---|---|
| 0 – 74% of threshold | **NORMAL — AWAKE** | 🟢 Green |
| 75 – 99% of threshold | **CAUTION — WARNING** | 🟡 Yellow |
| ≥ 100% of threshold | **ALERT — DROWSY** | 🔴 Red |

With the default threshold of 20 frames at ~30 FPS:
- **Green** → frames 0–14 (~normal blinking range)
- **Yellow** → frames 15–19 (~500 ms, eyes are clearly closing)
- **Red** → frame 20+ (~667 ms of sustained closure → alarm fires)

### 4. Web Dashboard
A Flask server streams the processed video and stats to a real-time browser dashboard via MJPEG and Server-Sent Events (SSE), accessible at `http://localhost:5050`.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Real-time EAR | Computed every frame at full webcam speed (~30 FPS) |
| 🔵 Face Oval Tracking | Draws the full face outline — green/yellow/red reflects current alert state |
| 🚦 3-Stage Alerting | Progressive warning before full alarm — reduces false-positive disruptions |
| 🌐 Web Dashboard | Live browser interface: EAR chart, status ring, event log, telemetry |
| 📊 EAR History Chart | Rolling 80-frame line chart showing EAR over time with threshold line |
| 🔊 Audio Alarm | Two-tone alarm fires only on full DROWSY state (not on caution) |
| ⚡ Flashing Banner | Red overlay banner when drowsiness is sustained |
| 📝 Driver Event Log | Timestamped log of CAUTION, DROWSINESS, and RECOVERED events |
| 🔢 Blink Counter | Counts total blinks across the session |
| 🎚️ Adaptive Threshold | EAR threshold adjustable live via dashboard slider |
| 👁️ Personal Calibration | One-click calibration to set threshold for individual eye geometry |
| 📡 Local Network Access | Dashboard reachable from other devices on the same network |

---

## 🚦 Alert Stages

The system transitions smoothly between three stages with no abrupt jumps:

### 🟢 Normal — Awake
- Face oval: **green**
- Status ring: green glow
- Header pill: **DRIVER ALERT**
- No audio

### 🟡 Caution — Warning
- Face oval: **yellow** (pulsing glow)
- Status ring: yellow glow with animation
- Header pill: **CAUTION**
- Video footer: *"Caution: Eyes Closing"*
- Log entry: `CAUTION` (yellow dot)
- No audio — this is a visual nudge only

### 🔴 Alert — Drowsy
- Face oval: **red**
- Status ring: red pulsing glow
- Header pill: **DROWSINESS DETECTED**
- Flashing red banner: *"DROWSINESS DETECTED — PLEASE TAKE A BREAK"*
- Audio alarm fires
- Log entry: `DROWSINESS` (red dot)

### 🟢 Recovered
After drowsy or caution states, once eyes reopen:
- All indicators return to green
- Log entry: `RECOVERED` (green dot) — recorded after both caution and full alert

---

## 📁 Project Structure

```
drowsiness-detector/
│
├── web_app.py            ← Flask server — main entry point
├── generate_alarm.py     ← Generates alarm.wav (run once)
├── alarm.wav             ← Alert sound (auto-generated)
│
├── utils/
│   ├── __init__.py
│   ├── ear.py            ← Eye Aspect Ratio formula
│   └── visualizer.py     ← OpenCV drawing: face oval, status bar, alert banner
│
├── templates/
│   └── index.html        ← Dashboard HTML
│
├── static/
│   ├── dashboard.css     ← Dark glassmorphism UI styles
│   └── dashboard.js      ← SSE client, chart rendering, status logic
│
├── logs/
│   └── drowsiness_log.csv  ← Session event log (auto-created)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- A working webcam

### Step 1 — Clone the Repository
```bash
git clone https://github.com/<your-username>/drowsiness-detector.git
cd drowsiness-detector
```

### Step 2 — Create a Virtual Environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate the Alarm Sound
```bash
python generate_alarm.py
```
This creates `alarm.wav` using only Python built-ins — no external tools needed.

---

## 🚀 Usage

### Start the Dashboard Server
```bash
python web_app.py
```

Then open your browser at **http://localhost:5050**

The dashboard is also accessible from other devices on the same local network at the IP address printed in the terminal (e.g. `http://192.168.x.x:5050`).

### With Custom Options
```bash
# Use camera index 1, stricter threshold, longer delay before alert
python web_app.py --cam 1 --ear 0.22 --frames 25

# Disable audio (visual alert only)
python web_app.py --no-sound
```

---

## 🖥️ Dashboard

The web dashboard provides a complete real-time monitoring interface:

| Panel | Contents |
|---|---|
| **Live Camera Feed** | MJPEG stream with face oval overlay and thin status bar |
| **EAR History Chart** | Rolling 80-frame line chart with threshold marker |
| **Status Ring** | Large circular indicator — green / yellow / red with glow animation |
| **Telemetry** | EAR value, threshold, blink count, FPS, session duration |
| **Calibrate Button** | 10-second calibration to set your personal EAR threshold |
| **Driver Log** | Timestamped list of CAUTION, DROWSINESS, and RECOVERED events |

### Personal Calibration
Click **"Calibrate for My Eyes"** and keep your eyes open naturally for 10 seconds. The system averages your open-eye EAR samples and sets the threshold to `mean × 0.75`, accounting for personal eye geometry (especially useful for people with naturally smaller or narrower eyes).

---

## 🎛️ Configuration

| Argument | Default | Description |
|---|---|---|
| `--cam` | `0` | Webcam index (0 = default camera) |
| `--ear` | `0.20` | Starting EAR threshold (overridden by calibration) |
| `--frames` | `20` | Consecutive frames below threshold before DROWSY fires |
| `--no-sound` | `False` | Disable audio alarm |

**Tuning tips:**
- If alerts trigger during normal blinks → increase `--frames` to `25–30`
- If detection is too slow → decrease `--frames` to `15`
- For small/narrow eyes → use the Calibrate button rather than manually lowering `--ear`
- The warning stage always fires at 75% of `--frames` (e.g. frame 15 of 20)

---

## 🔬 Technical Details

### Landmark Selection (MediaPipe FaceMesh)
MediaPipe provides 468 landmarks per frame. Six per eye are used for EAR:

| Eye   | Landmark Indices |
|-------|-----------------|
| Left  | `362, 385, 387, 263, 373, 380` |
| Right | `33, 160, 158, 133, 153, 144`  |

The face oval (36 boundary landmarks) is additionally rendered as a tracking outline.

### Architecture
The system runs two concurrent threads:

| Thread | Role |
|---|---|
| **Detection thread** (daemon) | Captures frames, runs MediaPipe, computes EAR, manages alert state, encodes JPEG |
| **Flask thread** (main) | Serves the dashboard, streams MJPEG via `/video_feed`, pushes SSE stats via `/stats_stream` |

Thread safety is managed by a single `threading.RLock` wrapping `DetectorState` — used as a context manager on every read/write.

### Why EAR Over Deep Learning?
- **Interpretable**: The formula has clear geometric meaning
- **Fast**: Runs in real-time on CPU — no GPU needed
- **Lightweight**: No model training required
- **Proven**: Published peer-reviewed research (Soukupová & Čech, 2016)

### Performance
- Runs at **25–30 FPS** on a mid-range CPU (Intel i5 / Ryzen 5)
- Memory footprint: ~150–200 MB (primarily MediaPipe)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, image processing, frame encoding |
| `mediapipe` | 468-point FaceMesh landmark detection |
| `numpy` | Numerical array operations |
| `flask` | Web server, MJPEG streaming, SSE endpoint |
| `scipy` | Euclidean distance for EAR calculation |
| `pygame` | Cross-platform audio alarm playback |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**ADVIK BANERJEE**  
Computer Vision — BYOP Submission  
VIT BHOPAL UNIVERSITY - 2026
