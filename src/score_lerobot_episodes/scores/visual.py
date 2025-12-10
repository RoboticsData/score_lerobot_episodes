import cv2
import numpy as np
import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from vlm import VLMInterface

##SCORING FRAME FUNCTIONS


def calculate_blur_score(gray: np.ndarray, max_var = 1000.0) -> float:
    """
    Calculate blur score using Laplacian variance.
    Higher score = sharper image.

    Args:
        gray: Grayscale image as numpy array

    Returns:
        Blur score (0-1, where 1 = sharp, 0 = blurry)
        Normalized based on typical variance ranges
    """
    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize to 0-1 range
    # Typical ranges: <100 = blurry, >500 = sharp
    # Using sigmoid-like normalization
    normalized = min(laplacian_var / max_var, 1.0)

    return float(normalized)

def calculate_darkness_score(gray: np.ndarray) -> float:
    """
    Calculate darkness score based on mean brightness.
    Higher score = brighter image.

    Args:
        gray: Grayscale image as numpy array

    Returns:
        Darkness score (0-1, where 1 = bright, 0 = dark)
    """
    # Calculate mean brightness
    mean_brightness = gray.mean()

    # Normalize to 0-1 range (0-255 -> 0-1)
    normalized = mean_brightness / 255.0

    return float(normalized)

def score_negative_visual_quality_opencv(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1-a  Sharpness (variance of Laplacian)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_penalty = max(0.0, min(1.0, 1 - fm/80.0))  # 0 good → 1 bad

    # 1-b  Exposure
    mean_intensity = gray.mean()
    if mean_intensity < 50:                           # too dark
        exposure_penalty = (50.0 - mean_intensity) / 50.0
    else:
        exposure_penalty = 0.0

    return max(blur_penalty, exposure_penalty)

##VIDEO SCORING FUNCTIONS

def score_visual_clarity(
    vp: str | pathlib.Path,
    sts,                       # unused but kept for signature compatibility
    acts,                      # unused but kept for signature compatibility
    vlm,                      # may be None
    task, nom,                # also unused here
    sample_every: int = 60
) -> float:
    cap = cv2.VideoCapture(str(vp))
    penalties, i = [], 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % sample_every:
            continue

        if vlm is not None and hasattr(vlm, "negative_visual_quality"):
            penalty = vlm.negative_visual_quality(frame)
        else:
            penalty = score_negative_visual_quality_opencv(frame)

        penalties.append(float(penalty))
    cap.release()
    return 0.0 if not penalties else max(0.0, 1.0 - np.mean(penalties))

if __name__ == '__main__':
    score = score_visual_clarity(
        vp='input_video.mp4',
        sts=[{}],            # no state needed
        vlm=VLMInterface(),
        task=None,
        nom=None,
    )

    # Basic sanity check
    assert 0.0 <= score <= 1.0, "score out of expected [0, 1] range"
    print('Visual score: ', score)
