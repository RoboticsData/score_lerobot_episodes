import cv2, numpy as np, pathlib

from .visual import score_visual_clarity
from .path import score_smoothness, score_path_efficiency, score_collision, score_joint_stability, score_gripper_consistency

def build_time_stats(states):
    """
    states – iterable of episode state-dicts, each containing key "t"
             (monotonically increasing timestamps, shape (N,))
    Returns  – dict with mean/std + Tukey–IQR fences for outlier tests
    """
    durs = [s["t"][-1] - s["t"][0] for s in states
            if "t" in s and len(s["t"]) > 1]
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

def score_task_success(vp, st, vlm, task, nom): return vlm.task_success(str(vp), task) if vlm is not None else 0.5

def score_runtime(vp, st, vlm, task, nom,
                  time_stats: dict | None = None,
                  outlier_penalty: float = 0.):
    """
    • If `time_stats` is supplied, an episode whose length is an outlier
      (see helper above) gets `outlier_penalty` (default 0 → fail hard).
    • Otherwise we keep the original exponential decay:
        exp(- duration / nom)
      If `nom` is <=0, we fall back to the *global* mean duration.
    """
    t = st["t"]
    duration = t[-1] - t[0]

    # 1-a  Outlier check  ------------------------------------------------
    if time_stats and is_time_outlier(duration, time_stats):
        return outlier_penalty

    # 1-b  Nominal time  -------------------------------------------------
    if not nom or nom <= 0:
        nom = time_stats["mean"] if time_stats else duration

    return float(np.exp(-duration / nom))
