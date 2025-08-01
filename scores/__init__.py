import cv2, numpy as np, pathlib

from .visual import score_visual_clarity
from .path import score_smoothness, score_path_efficiency, score_collision, score_joint_stability, score_gripper_consistency

def build_time_stats(states):
    """
    states – iterable of episode state-dicts, each containing key "t"
             (monotonically increasing timestamps, shape (N,))
    Returns  – dict with mean/std + Tukey–IQR fences for outlier tests
    """
    durs = [s[-1]["t"] - s[0]["t"] for s in states]
    if not durs:                                 # fallback if nothing valid
        return {"mean": 0., "std": 0., "q1": 0., "q3": 0., "iqr": 0.}

    durs = np.asarray(durs, dtype=float)
    q1, q3 = np.percentile(durs, (25, 75))
    return {
        "mean": durs.mean(),
        "std" : durs.std(ddof=0),
        "q1"  : q1,
        "q3"  : q3,
        "iqr" : q3 - q1,
    }

def is_time_outlier(duration, stats, mode="iqr", z_thresh=3.):
    """
    mode = "iqr"  → Tukey fence: 1.5×IQR outside [Q1, Q3]
    mode = "z"    → |Z| > z_thresh
    """
    if mode == "iqr":
        lo = stats["q1"] - 1.5*stats["iqr"]
        hi = stats["q3"] + 1.5*stats["iqr"]
        return duration < lo or duration > hi
    else:  # z-score
        if stats["std"] == 0.:            # avoid div-by-zero
            return False
        z = abs(duration - stats["mean"]) / stats["std"]
        return z > z_thresh

def score_task_success(vp, sts, vlm, task, nom): return vlm.task_success(str(vp), task) if vlm is not None else 0.5

def score_runtime(vp, sts, vlm, task, nom,
                  time_stats: dict | None = None,
                  outlier_penalty: float = 0.0):
    """
    • If `time_stats` is supplied, an episode whose length is an outlier
      (see helper above) gets `outlier_penalty` (default 0 → fail hard).
      If `nom` is <=0, we fall back to the *global* mean duration.
    """
    timestamps = np.array([st["t"] for st in sts])
    duration = timestamps[-1] - timestamps[0]

    print(time_stats, timestamps, duration)

    # 1-a  Outlier check  ------------------------------------------------
    if time_stats and is_time_outlier(duration, time_stats):
        return outlier_penalty

    return 1
