"""
generate_alarm.py — Alarm Sound Generator
==========================================
Generates a two-tone alarm WAV file (alarm.wav) using only Python
built-in modules — no external dependencies required.

Run once before starting the detector:
    python generate_alarm.py
"""

import wave
import struct
import math
import os

OUTPUT_FILE = "alarm.wav"
SAMPLE_RATE = 44100   # Hz
DURATION    = 1.2     # seconds total
FREQ_HIGH   = 880     # A5 — high tone (Hz)
FREQ_LOW    = 440     # A4 — low tone  (Hz)
VOLUME      = 0.7     # 0.0 – 1.0


def generate_tone(freq: float, duration: float, sample_rate: int,
                  volume: float) -> list:
    """Return PCM samples for a sine wave at the given frequency."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # Sine wave with a short fade-in / fade-out to avoid clicks
        fade   = min(1.0, min(t, duration - t) / 0.02)  # 20 ms fade
        sample = volume * fade * math.sin(2 * math.pi * freq * t)
        samples.append(int(sample * 32767))
    return samples


def write_wav(filename: str, samples: list, sample_rate: int) -> None:
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        packed = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(packed)


def main():
    half = DURATION / 2
    high_samples = generate_tone(FREQ_HIGH, half, SAMPLE_RATE, VOLUME)
    low_samples  = generate_tone(FREQ_LOW,  half, SAMPLE_RATE, VOLUME)
    all_samples  = high_samples + low_samples

    write_wav(OUTPUT_FILE, all_samples, SAMPLE_RATE)
    print(f"[OK] Alarm sound saved -> {os.path.abspath(OUTPUT_FILE)}")
    print(f"    Duration : {DURATION}s | Sample rate : {SAMPLE_RATE} Hz")
    print(f"    Tones    : {FREQ_HIGH} Hz -> {FREQ_LOW} Hz")


if __name__ == "__main__":
    main()
