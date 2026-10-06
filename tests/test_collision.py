import numpy as np

from score_lerobot_episodes.scores.path import score_collision

FPS = 30


def _states(q, fps=FPS):
    q = np.asarray(q, dtype=float).reshape(len(q), -1)
    return [{"q": row, "t": i / fps} for i, row in enumerate(q)]


def _reach_with_holds(hold_s=2.0, reach_s=2.0, fps=FPS):
    """Hold, smooth cosine reach from 0 to 1 rad, hold again."""
    t = np.arange(0, 2 * hold_s + reach_s, 1 / fps)
    s = np.clip((t - hold_s) / reach_s, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * s))


def test_smooth_reach_with_holds_has_no_spikes():
    # The joint is still for more than half the episode, so its median
    # |accel| is 0. A smooth reach must not be read as a collision.
    q = _reach_with_holds()
    assert score_collision(None, _states(q), None, None, None, None) == 1.0


def test_still_joint_next_to_moving_joint_has_no_spikes():
    q = _reach_with_holds()
    both = np.stack([q, np.zeros_like(q)], axis=1)
    assert score_collision(None, _states(both), None, None, None, None) == 1.0


def test_abrupt_stop_is_flagged():
    # Same reach, but the joint stops dead at its fastest point.
    q = _reach_with_holds()
    mid = len(q) // 2
    q[mid:] = q[mid]
    score = score_collision(None, _states(q), None, None, None, None)
    assert 0.0 < score < 1.0


def test_constant_state_scores_one():
    q = np.full((60, 3), 0.25)
    assert score_collision(None, _states(q), None, None, None, None) == 1.0
