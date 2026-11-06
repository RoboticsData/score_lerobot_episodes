"""Regression tests for issue #48 (score_collision zero-median bug).

Run: pytest tests/test_collision.py -q
"""
import numpy as np
import pytest

from score_lerobot_episodes.scores.path import collision_spike_mask, score_collision

FPS = 30


def _states(q, fps=FPS):
    q = np.asarray(q, dtype=float).reshape(len(q), -1)
    return [{"q": row, "t": i / fps} for i, row in enumerate(q)]


def _reach_with_holds(hold_s=2.0, reach_s=2.0, fps=FPS):
    """Hold, smooth cosine reach from 0 to 1 rad, hold again."""
    t = np.arange(0, 2 * hold_s + reach_s, 1 / fps)
    s = np.clip((t - hold_s) / reach_s, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * s))


def _pad_still(q, n, fps=FPS):
    """Prepend n identical frames (operator waiting before the motion)."""
    return np.vstack([np.repeat(q[:1], n, axis=0), q])


# ── Cases from PR #49 ──────────────────────────────────────────────

def test_smooth_reach_with_holds_has_no_spikes():
    q = _reach_with_holds()
    assert score_collision(None, _states(q), None, None, None, None) == 1.0


def test_still_joint_next_to_moving_joint_has_no_spikes():
    q = _reach_with_holds()
    both = np.stack([q, np.zeros_like(q)], axis=1)
    assert score_collision(None, _states(both), None, None, None, None) == 1.0


def test_abrupt_stop_is_flagged():
    q = _reach_with_holds()
    mid = len(q) // 2
    q[mid:] = q[mid]
    score = score_collision(None, _states(q), None, None, None, None)
    assert 0.0 < score < 1.0


def test_constant_state_scores_one():
    q = np.full((60, 3), 0.25)
    assert score_collision(None, _states(q), None, None, None, None) == 1.0


# ── New: root-cause regression & score invariance ───────────────────

def test_quantized_joint_with_majority_still_frames():
    """ALOHA-like signal: encoder steps of constant size, still > 50% of frames.

    The old median-over-all-frames threshold is 0 here -> 99%+ false spikes.
    """
    steps = 120
    q = np.zeros((300, 4))
    for i in range(90, 90 + steps):
        q[i, 0] = (i - 89) * 0.0015          # constant encoder step
    s = _states(q, fps=50)
    assert score_collision(None, s, None, None, None, None) > 0.99


@pytest.mark.parametrize("pad", [0, 20, 100, 400])
def test_idle_padding_does_not_change_detected_spikes(pad):
    """A score must not depend on how long the operator waited before moving.

    mean(|accel|) over all frames (PR #49) shrinks with padding, so its
    threshold shrinks and normal motion starts getting flagged. Active-frame
    statistics are invariant to padding by construction.
    """
    q = _reach_with_holds()
    raw_mask = collision_spike_mask(_states(q))
    padded = _states(_pad_still(q.reshape(-1, 1), pad))
    padded_mask = collision_spike_mask(padded)
    # compare only the original action segment (last len(raw_mask) accel frames)
    assert np.array_equal(raw_mask, padded_mask[-len(raw_mask):])


def test_impact_velocity_reversal_is_detected():
    """A violent velocity reversal during motion must produce a spike."""
    q = _reach_with_holds().reshape(-1, 1)
    t0 = len(q) // 2
    step = q[t0, 0] - q[t0 - 1, 0]
    assert abs(step) > 1e-4
    impacted = q.copy()
    impacted[t0:, 0] += -20 * step            # 20x step velocity reversal
    clean_mask = collision_spike_mask(_states(q))
    hit_mask = collision_spike_mask(_states(impacted))
    assert clean_mask.sum() == 0
    assert hit_mask[max(0, t0 - 3):t0 + 3].any()


def test_short_episode_does_not_crash():
    for n in (1, 2):
        s = _states(np.full((n, 2), 0.3))
        assert score_collision(None, s, None, None, None, None) == 1.0
        assert len(collision_spike_mask(s)) == n
