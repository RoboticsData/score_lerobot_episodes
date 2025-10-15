import numpy as np

def rms(x, axis=None): return float(np.sqrt(np.mean(np.square(x), axis=axis)))

def score_smoothness(vp, sts, vlm, task, nom, *, k: float = 1000.0):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')
    accel = np.diff(states, 2, 0) / np.diff(timestamps)[:-1, None]**2
    scores = float(np.exp(-rms(accel) / k))
    return np.mean(scores)

def score_path_efficiency(vp, sts, vlm, task, nom):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')
    # Joint-space path length
    path = np.sum(np.linalg.norm(np.diff(states, axis=0), axis=1))

    # Joint-space straight-line distance
    straight = np.linalg.norm(states[-1] - states[0])

    scores = 0. if path < 1e-6 else float(np.clip(straight / path, 0., 1.))
    return np.mean(scores)

def score_idle_velocity(vp, sts, vlm, task, nom):
    threshold = 0.1
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    """Detect idle based on low velocity."""
    velocities = np.diff(states, axis=0) / np.diff(timestamps)[:, None]
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    idle_mask = velocity_magnitude < threshold

    #idle_time = np.sum(idle_mask * np.diff(timestamps))
    idle_ratio = np.mean(idle_mask)

    return 1-idle_ratio

def score_collision(vp, sts, vlm, task, nom):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')

    # Compute second derivative (acceleration proxy) in joint space
    accel = np.diff(states, n=2, axis=0) / (np.diff(timestamps)[:-1, None] ** 2)

    # Detect "spikes" in joint-space acceleration
    threshold = 15.0 * np.median(np.abs(accel), axis=0, keepdims=True)
    spike_mask = np.any(np.abs(accel) > threshold, axis=1)

    # Score is high if few or no spikes
    spike_ratio = np.mean(spike_mask)
    scores = max(0.0, 1.0 - spike_ratio)
    return np.mean(scores)

def score_joint_stability(vp, sts, vlm, task, nom):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')

    # Consider the final 2 seconds of the episode
    mask = timestamps >= timestamps[-1] - 2.0
    if not np.any(mask):
        return 0.0

    # Standard deviation of joint angles in that window
    final_state = states[mask]
    joint_std = np.std(final_state, axis=0).mean()

    # Lower std = more stable. Use exponential decay scoring
    scores = float(np.exp(-joint_std / 0.05))  # adjust denominator for sensitivity
    return np.mean(scores)

def score_gripper_consistency(vp, sts, vlm, task, nom):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')

    grip = np.array([st.get("grip") for st in sts])
    #prob = vlm.task_success(str(vp), "The robot is holding the object.")
    if np.any(grip) is None: return 0.5
    agree = (grip.astype(bool) == (grip >= 0.5)).mean()
    scores = max(0., min(1., (agree - 0.1) / 0.9))
    return np.mean(scores)
