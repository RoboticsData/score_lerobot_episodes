import cv2
import numpy as np
import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from vlm import VLMInterface

def score_visual_clarity(
    vp: str | pathlib.Path,
    st,                       # unused but kept for signature compatibility
    vlm,                      # may be None
    task, nom,                # also unused here
    sample_every: int = 20
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


def score_negative_visual_quality_opencv(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1-a  Sharpness (variance of Laplacian)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_penalty = max(0.0, min(1.0, (80.0 - fm) / 80.0))  # 0 good → 1 bad

    # 1-b  Exposure
    mean_intensity = gray.mean()
    if mean_intensity < 50:                           # too dark
        exposure_penalty = (50.0 - mean_intensity) / 50.0
    elif mean_intensity > 200:                        # too bright
        exposure_penalty = (mean_intensity - 200.0) / 55.0
    else:
        exposure_penalty = 0.0

    return max(blur_penalty, exposure_penalty)

if __name__ == '__main__':
    score = score_visual_clarity(
        vp='input_video.mp4',
        st={},            # no state needed
        vlm=VLMInterface(),
        task=None,
        nom=None,
    )

    # Basic sanity check
    assert 0.0 <= score <= 1.0, "score out of expected [0, 1] range"
    print('Visual score: ', score)