import numpy as np
from scipy.fft import rfft
from scipy.interpolate import interp1d

from score_lerobot_episodes.util import VideoSegment

def rms(x, axis=None): return float(np.sqrt(np.mean(np.square(x), axis=axis)))

def score_smoothness(video_segment: VideoSegment, sts, acts, vlm, task, nom, *, k: float = 1000.0):
    states = np.array([st.get("q") for st in sts])
    timestamps = np.array([st["t"] for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')
    accel = np.diff(states, 2, 0) / np.diff(timestamps)[:-1, None]**2
    scores = float(np.exp(-rms(accel) / k))
    return np.mean(scores)

def score_path_efficiency(video_segment: VideoSegment, sts, acts, vlm, task, nom):
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

def score_idle_velocity(video_segment: VideoSegment, sts, acts, vlm, task, nom):
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

def score_collision(video_segment: VideoSegment, sts, acts, vlm, task, nom):
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

def score_joint_stability(video_segment: VideoSegment, sts, acts, vlm, task, nom):
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

def score_gripper_consistency(video_segment: VideoSegment, sts, acts, vlm, task, nom):
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

def score_actuator_saturation(video_segment: VideoSegment, sts, acts, vlm, task, nom, *, threshold_deg: float = 7):
    states = np.array([st.get("q") for st in sts])
    if (states == None).any():
        raise ValueError('Invalid state vector')

    # Ensure we have matching dimensions
    # actions[t] should correspond to transition from states[t] to states[t+1]
    actions = np.array(acts)
    assert(len(actions) == len(states))

    # Compute |a_t - s_{t+1}| for each timestep
    action_state_diff = np.abs(actions[:-1] - states[1:])

    # Check what fraction of time each joint exceeds threshold
    # TODO: This assumes that the action space is in degrees and that it is int the same format as the state space.
    saturation_mask = action_state_diff > threshold_deg

    # Compute saturation ratio (fraction of timesteps where any joint is saturated)
    saturation_ratio = np.mean(np.any(saturation_mask, axis=1))

    # Score: exponential decay based on saturation ratio
    # Rationale: We want to penalize more if there is any saturation at all.
    scores = float(np.exp(-4.0 * saturation_ratio))
    return scores

def compute_sparc(sts, cutoff_freq=10.0, pad_factor=5):
    """
    Computes the Spectral Arc Length (SPARC) for a movement trajectory.
    
    Key Parameters:
    - sts: list of state dictionaries
        Each dictionary contains a "q" key with the joint angles.
        Each dictionary contains a "t" key with the timestamp.
    - cutoff_freq: max frequency to consider (default 10Hz for human-like motion)
    - pad_factor: multiplier for zero-padding to increase FFT resolution
    
    Returns:
    - sparc_score: The smoothness metric (closer to 0 is smoother)
    """
    qpos = np.array([st.get("q") for st in sts])
    times = np.array([st["t"] for st in sts])
    if qpos.ndim == 1:
        qpos = qpos.reshape(-1, 1)
    assert qpos.ndim == 2, "qpos must be 2D"

    # Calculate the average sampling rate from the timestamps 
    sampling_rate = 1 / np.mean(np.diff(times))

    # 1. Calculate Velocity (Angular Rate)
    # Using gradient to maintain the same array length as input
    velocity = np.gradient(qpos, times, axis=0)

    # 2. Pre-processing: Remove DC offset (mean) and apply zero-padding
    # This improves the frequency resolution (df)
    n_points = len(velocity)
    n_fft = int(2**np.ceil(np.log2(n_points * pad_factor)))

    # 3. Compute Magnitude Spectrum
    # We use the FFT of the velocity profile
    v_fft = rfft(velocity, axis=0, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1/sampling_rate)
    
    # Only keep the positive frequencies up to our cutoff    
    positive_indices = np.where((freqs >= 0) & (freqs <= cutoff_freq))[0]
    mag_spectrum = np.abs(v_fft[positive_indices])
    selected_freqs = freqs[positive_indices]
    
    # 4. Normalize the Spectrum
    # Scale so the maximum amplitude is 1.0
    max_vals = np.max(mag_spectrum, axis=0)
    safe_max_vals = np.where(max_vals == 0, 1, max_vals)
    norm_spectrum = mag_spectrum / safe_max_vals

    def compute_arc_length(spectrum, freqs):
        # 5. Adaptive Cutoff
        # Find the last frequency where the amplitude is > 0.05 of the peak
        # This prevents the "tail" of the noise from dominating the arc length
        threshold = 0.05
        significant_indices = np.where(spectrum > threshold)[0]
        if len(significant_indices):
            index_limit = significant_indices[-1]+1
        else:
            index_limit = len(spectrum)

        final_spectrum = spectrum[:index_limit]
        final_freqs = freqs[:index_limit]

        # 6. Calculate Arc Length
        # We calculate the distance between points in the (freq, amplitude) plane
        # Formula: sum(sqrt( (delta_freq)^2 + (delta_amplitude)^2 ))
        df = np.diff(final_freqs, axis=0)
        da = np.diff(final_spectrum, axis=0)

        # Normalizing the frequency axis by the cutoff frequency (fc) as per SPARC definition
        # This makes the metric dimensionless and independent of movement duration
        fc = final_freqs[-1]
        if fc == 0: return 0.0

        # Calculate arc length
        # Note: SPARC is usually expressed as a negative value (closer to 0 is smoother)
        return -np.sum(np.sqrt((df / fc)**2 + da**2))

    arc_lengths = []
    for i in range(norm_spectrum.shape[1]):
        arc_lengths.append(compute_arc_length(norm_spectrum[:, i], selected_freqs))

    return np.array(arc_lengths)


def score_sparc(video_segment: VideoSegment, sts, acts, vlm, task, nom, beta: float = 0.1, offset: float = 1.0):
    """
    Scores the SPARC metric for a movement trajectory.
    
    Parameters:
    - sts: list of state dictionaries
        Each dictionary contains a "q" key with the joint angles.
        Each dictionary contains a "t" key with the timestamp.
    - beta: float, the exponent for the exponential decay
    - offset: float, the offset for the SPARC score
    
    Returns:
    - sparc_scores: numpy array of SPARC scores for each joint angle trajectory
    """

    arc_lengths = compute_sparc(sts)
    sparc_scores = np.minimum(np.exp(beta * (arc_lengths + offset)), 1.0)
    return sparc_scores