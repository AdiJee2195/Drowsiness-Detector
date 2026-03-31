"""
ear.py — Eye Aspect Ratio (EAR) Calculation
============================================
Implements the EAR formula from:
    Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks"
    (CVWW 2016)

Formula:
    EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)

Where p1–p6 are the 6 eye landmark coordinates (horizontal endpoints +
two pairs of vertical endpoints).
"""

import numpy as np
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye: list) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Args:
        eye: list of 6 (x, y) coordinate tuples in order:
             [left_corner, top_left, top_right, right_corner, bot_right, bot_left]

    Returns:
        float: EAR value (~0.3 open, ~0.0 closed)
    """
    # Vertical euclidean distances
    A = dist.euclidean(eye[1], eye[5])  # top-left  <-> bot-left
    B = dist.euclidean(eye[2], eye[4])  # top-right <-> bot-right

    # Horizontal euclidean distance
    C = dist.euclidean(eye[0], eye[3])  # left corner <-> right corner

    ear = (A + B) / (2.0 * C)
    return ear


def average_ear(left_eye: list, right_eye: list) -> float:
    """
    Compute the average EAR across both eyes.

    Args:
        left_eye:  6 landmark coords for the left eye
        right_eye: 6 landmark coords for the right eye

    Returns:
        float: average EAR
    """
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
