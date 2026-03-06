import cv2
import numpy as np
import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from score_lerobot_episodes.vlm import VLMInterface
from score_lerobot_episodes.util import VideoSegment, iterate_frames_in_range

##SCORING FRAME FUNCTIONS

def calculate_blur_score(gray: np.ndarray, max_var: float = 1000.0) -> float:
    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Typical ranges: <100 = blurry, >500 = sharp
    normalized = min(laplacian_var / max_var, 1.0) # Normalize to 0-1 range

    return float(normalized)

def calculate_darkness_score(gray: np.ndarray, max_brightness: float = 255.0) -> float:
    mean_brightness = gray.mean() # Calculate mean brightness
    # Normalize to 0-1 range (0-255 -> 0-1)
    normalized = mean_brightness / max_brightness
    return float(normalized)

def calculate_contrast_score(gray: np.ndarray, max_std: float = 80.0) -> float:
    # Calculate standard deviation as measure of contrast
    std_dev = gray.std()

    # Normalize to 0-1 range
    # Typical ranges: <20 = low contrast, >50 = high contrast
    normalized = min(std_dev / max_std, 1.0)

    return float(normalized)

def calculate_exposure_score(gray: np.ndarray, use_center_weighting: bool = True) -> float:
    """
    Computes an exposure score (0.0 to 1.0) where 1.0 is perfectly exposed.
    Uses optional center-weighting
    """

    h, w = gray.shape

    if use_center_weighting:
        # Create a 2D Gaussian Center-Weighting Mask
        # The intuition is that the center of the image is usually more important.
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        x_grid, y_grid = np.meshgrid(x, y)
        kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * 0.5**2))
        kernel = kernel / np.sum(kernel) # Normalize mask

        # Calculate Weighted Mean
        weighted_mean = np.sum(gray * kernel)
    else:
        weighted_mean = gray.mean()

    # Target Exposure (18% Gray in sRGB is ~117-128)
    target = 127.5
    exposure_err = abs(weighted_mean - target) / target # 0 is perfect, 1 is total black/white


    # Final Score Calculation
    # We invert the error so that higher = better.
    return max(0, 1.0 - exposure_err)

def score_negative_visual_quality_opencv(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1-a  Sharpness (variance of Laplacian)
    blur_score = calculate_blur_score(gray, max_var = 80.0)
    blur_penalty = max(0.0, min(1.0, 1 - blur_score))  # 0 = sharp → 1 = blurry

    # 1-b  Exposure
    dark_score = calculate_darkness_score(gray, max_brightness = 50.0)
    exposure_penalty = 1.0 - dark_score 

    return max(blur_penalty, exposure_penalty)


##VIDEO SCORING FUNCTIONS

def score_visual_clarity(
    video_segment: VideoSegment,
    sts,                       # unused but kept for signature compatibility
    acts,                      # unused but kept for signature compatibility
    vlm,                      # may be None
    task, nom,                # also unused here
    sample_every: int = 60
) -> float:

    penalties, i = [], 0
    for frame in iterate_frames_in_range(video_segment):
        i += 1
        if i % sample_every:
            continue

        if vlm is not None and hasattr(vlm, "negative_visual_quality"):
            penalty = vlm.negative_visual_quality(frame)
        else:
            penalty = score_negative_visual_quality_opencv(frame)

        penalties.append(float(penalty))

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
