# 🚗 Real-Time Driver Drowsiness Detection System

> A Computer Vision project that monitors a driver's eyes in real-time using a webcam and triggers an audio + visual alert when prolonged eye closure (drowsiness) is detected.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Controls](#-controls)
- [Output & Logs](#-output--logs)
- [Technical Details](#-technical-details)
- [Dependencies](#-dependencies)

---

## 🎯 Problem Statement

Drowsy driving is a leading cause of road accidents worldwide. According to NHAI data, fatigue-related crashes account for a significant proportion of highway fatalities in India. Drivers often fall asleep without realizing it — making **early detection critical**.

This system provides a real-time, non-intrusive computer vision solution: it watches the driver's eyes through a webcam and fires an alert the moment sustained eye closure is detected, giving the driver time to pull over safely.

---

## 🧠 How It Works

The system uses a three-stage pipeline:

```
Webcam Frame → Face Landmark Detection → EAR Calculation → Alert Logic
```

### 1. Face Landmark Detection
**MediaPipe FaceMesh** detects 468 facial landmarks in each frame. Six specific landmarks per eye are extracted — chosen to match the EAR formula.

### 2. Eye Aspect Ratio (EAR)
The EAR is a ratio that quantifies how open the eye is:

```
        ||p2 - p6|| + ||p3 - p5||
EAR =  ───────────────────────────
              2 × ||p1 - p4||
```

Where `p1–p6` are the six eye landmark coordinates.

| State        | EAR Value  |
|--------------|------------|
| Eyes open    | ~0.30      |
| Eyes blinking| ~0.15      |
| Eyes closed  | ~0.0       |

📌 Reference: *Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", CVWW 2016*

### 3. Alert Logic
- If EAR stays **below 0.25** for **20+ consecutive frames** → **DROWSY**
- A flashing banner appears on screen and an audio alarm sounds
- Events are logged to `logs/drowsiness_log.csv`

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Real-time EAR | Computed every frame at full webcam speed |
| 👁️ Eye Landmark Overlay | Draws convex hulls around both eyes with color-coded status |
| 📊 HUD Panel | Shows EAR value, EAR bar, blink count, FPS, charge bar, and status |
| 🔊 Audio Alarm | Two-tone alarm (either via pygame or Windows winsound) |
| ⚡ Flashing Banner | Red warning banner when drowsiness is sustained |
| 📝 CSV Logging | Every alert start/end is timestamped and saved |
| 🔢 Blink Counter | Tracks total blinks in the session |
| ⌨️ Keyboard Controls | Quit, reset counter, toggle sound — all live |
| 🎛️ CLI Arguments | Customize threshold, frame count, camera, and alarm |

---

## 📁 Project Structure

```
drowsiness-detector/
│
├── detector.py           ← Main script — run this
├── generate_alarm.py     ← Generates alarm.wav (run once)
├── alarm.wav             ← Alert sound (auto-generated)
│
├── utils/
│   ├── __init__.py
│   ├── ear.py            ← Eye Aspect Ratio formula
│   └── visualizer.py     ← All OpenCV drawing functions
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

### Basic Run (default settings)
```bash
python detector.py
```

### With Custom Options
```bash
# Use camera index 1, stricter threshold, longer delay before alert
python detector.py --cam 1 --ear 0.22 --frames 25

# Disable audio (visual alert only)
python detector.py --no-sound

# Use a custom alarm file
python detector.py --alarm my_alarm.wav
```

---

## 🎛️ Configuration

| Argument | Default | Description |
|---|---|---|
| `--cam` | `0` | Webcam index (0 = default camera) |
| `--ear` | `0.25` | EAR threshold (lower = stricter) |
| `--frames` | `20` | Frames below threshold before alert fires |
| `--no-sound` | `False` | Disable audio alarm |
| `--alarm` | `alarm.wav` | Path to custom alarm WAV file |

**Tuning tips:**
- If alerts trigger during normal blinks → increase `--frames` to `25–30`
- If detection is too slow → decrease `--frames` to `15`
- If wearing glasses causes misdetection → decrease `--ear` to `0.22`

---

## ⌨️ Controls

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit the program |
| `R` | Reset the blink counter |
| `S` | Toggle sound on/off |

---

## 📊 Output & Logs

Every drowsiness event is automatically logged to `logs/drowsiness_log.csv`:

```csv
timestamp,event,ear_value
2025-04-01 10:14:32,DROWSY_START,0.1823
2025-04-01 10:14:38,DROWSY_END,0.2904
2025-04-01 10:22:07,DROWSY_START,0.1511
```

The terminal also prints real-time status messages:
```
⚠  ALERT  Drowsiness detected at 10:14:32  EAR=0.182
✓  OK     Driver alert again at 10:14:38
```

---

## 🔬 Technical Details

### Landmark Selection (MediaPipe FaceMesh)
MediaPipe provides 468 landmarks. We use 6 per eye specifically chosen to align with the EAR formula:

| Eye   | Landmark Indices |
|-------|-----------------|
| Left  | `362, 385, 387, 263, 373, 380` |
| Right | `33, 160, 158, 133, 153, 144`  |

### Why EAR Over Deep Learning?
- **Interpretable**: The formula has clear geometric meaning
- **Fast**: Runs in real-time on CPU — no GPU needed
- **Lightweight**: No model training required
- **Proven**: Published peer-reviewed research

### Performance
- Runs at **25–30 FPS** on a mid-range CPU (Intel i5 / Ryzen 5)
- Memory footprint: ~150–200 MB (primarily MediaPipe)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, image processing, rendering |
| `mediapipe` | 468-point FaceMesh landmark detection |
| `numpy` | Numerical array operations |
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
