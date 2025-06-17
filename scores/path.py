import numpy as np

def rms(x, axis=None): return float(np.sqrt(np.mean(np.square(x), axis=axis)))

def score_smoothness(vp, st, vlm, task, nom):
    q, t = st.get("q"), st["t"]
    if q is None: return 0.
    accel = np.diff(q, 2, 0) / np.diff(t)[:-1, None]**2
    score = float(np.exp(-rms(accel) / 10.))
    return score

def score_path_efficiency(vp, st, vlm, task, nom):
    q = st.get("q")
    if q is None or len(q) < 2: return 0.

    # Joint-space path length
    path = np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))

    # Joint-space straight-line distance
    straight = np.linalg.norm(q[-1] - q[0])

    score = 0. if path < 1e-6 else float(np.clip(straight / path, 0., 1.))
    return score

def score_collision(vp, st, vlm, task, nom):
    q, t = st.get("q"), st.get("t")
    if q is None or t is None or len(q) < 3:
        return 1.0  # assume no collision if insufficient data

    # Compute second derivative (acceleration proxy) in joint space
    accel = np.diff(q, n=2, axis=0) / (np.diff(t)[:-1, None] ** 2)

    # Detect "spikes" in joint-space acceleration
    threshold = 3.0 * np.median(np.abs(accel), axis=0, keepdims=True)
    spike_mask = np.any(np.abs(accel) > threshold, axis=1)

    # Score is high if few or no spikes
    spike_ratio = np.mean(spike_mask)
    score = max(0.0, 1.0 - spike_ratio)
    return score

def score_joint_stability(vp, st, vlm, task, nom):
    q, t = st.get("q"), st.get("t")
    if q is None or t is None or len(q) < 2:
        return 0.0

    # Consider the final 2 seconds of the episode
    mask = t >= t[-1] - 2.0
    if not np.any(mask):
        return 0.0

    # Standard deviation of joint angles in that window
    q_final = q[mask]
    joint_std = np.std(q_final, axis=0).mean()

    # Lower std = more stable. Use exponential decay scoring
    score = float(np.exp(-joint_std / 0.05))  # adjust denominator for sensitivity
    return score

def score_gripper_consistency(vp, st, vlm, task, nom):
    grip = st.get("grip")
    #prob = vlm.task_success(str(vp), "The robot is holding the object.")
    if grip is None: return 0.5
    agree = (grip.astype(bool) == (grip >= 0.5)).mean()
    score = max(0., min(1., (agree - 0.1) / 0.9))
    return score
