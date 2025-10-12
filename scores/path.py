import numpy as np

def rms(x, axis=None): return float(np.sqrt(np.mean(np.square(x), axis=axis)))

def score_smoothness(vp, sts, acts, vlm, task, nom, *, k: float = 1000.0):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')
    accel = np.diff(states, 2, 0) / np.diff(timestamps)[:-1, None]**2
    scores = float(np.exp(-rms(accel) / k))
    return np.mean(scores)

def score_path_efficiency(vp, sts, acts, vlm, task, nom):
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

def score_collision(vp, sts, acts, vlm, task, nom):
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

def score_joint_stability(vp, sts, acts, vlm, task, nom):
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

def score_gripper_consistency(vp, sts, acts, vlm, task, nom):
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

def score_actuator_saturation(vp, sts, acts, vlm, task, nom, *, threshold_deg: float = 7):
    """
    Score based on actuator saturation using |a_t - s_{t+1}|.
    Large differences between commanded actions and resulting states indicate
    actuators operating near their limits. Lower scores indicate more saturation.

    Args:
        threshold_deg: Threshold for |a_t - s_{t+1}|. Differences above this
                  are considered saturated (default: 10 deg)

    Returns:
        Score in [0, 1] where 1 = no saturation, 0 = fully saturated
    """
    states = np.array([st.get("q") for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')

    # Convert actions to numpy array
    actions = np.array(acts)

    # Ensure we have matching dimensions
    # actions[t] should correspond to transition from states[t] to states[t+1]
    assert(len(actions) == len(states))

    # Compute |a_t - s_{t+1}| for each timestep
    action_state_diff = np.abs(actions[:-1] - states[1:])

    # Check what fraction of time each joint exceeds threshold
    saturation_mask = action_state_diff > threshold_deg

    # Compute saturation ratio (fraction of timesteps where any joint is saturated)
    saturation_ratio = np.mean(np.any(saturation_mask, axis=1))

    # Score: exponential decay based on saturation ratio
    # Rationale: We want to penalize more if there is any saturation at all.
    scores = float(np.exp(-4.0 * saturation_ratio))
    return scores
