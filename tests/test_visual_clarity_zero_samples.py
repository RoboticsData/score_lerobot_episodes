"""
Regression tests for issue #30: a segment that cannot be measured must not
resolve to a score.

score_visual_clarity previously returned 0.0 whenever it collected no
samples. 0.0 is the worst achievable clarity, and the value feeds the
weighted aggregate that decides whether an episode is kept, so "nothing was
measured" was indistinguishable from "this footage is unusable".

These tests stub out frame iteration, so they need no video files, no codecs
and no ffmpeg binary.
"""

import numpy as np
import pytest

from score_lerobot_episodes.scores import visual as visual_module
from score_lerobot_episodes.scores.visual import score_visual_clarity
from score_lerobot_episodes.util import FrameDecodeError, VideoSegment


def _sharp_frame(size: int = 32) -> np.ndarray:
    """
    A high contrast checkerboard, so the frame scores well on its own merits.
    A flat frame would not work here: its Laplacian variance is zero, which
    the blur term correctly reads as maximally blurry and scores 0.0. That is
    the score these tests need to distinguish from an unmeasured segment.
    """
    row = np.indices((size, size)).sum(axis=0) % 2
    return np.repeat((row * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _segment() -> VideoSegment:
    return VideoSegment(video_path="stub.mp4", from_timestamp=0.0, to_timestamp=1.0)


def _score_over(monkeypatch, frame_count: int, **kwargs) -> float:
    frames = [_sharp_frame() for _ in range(frame_count)]
    monkeypatch.setattr(
        visual_module, "iterate_frames_in_range", lambda segment, **_: iter(frames)
    )
    return score_visual_clarity(
        _segment(), sts=None, acts=None, vlm=None, task=None, nom=None, **kwargs
    )


@pytest.mark.parametrize("frame_count", [1, 10, 59])
def test_segment_shorter_than_sample_interval_is_still_scored(monkeypatch, frame_count):
    """
    A segment shorter than sample_every decodes fine but used to sample no
    frames, because the index was tested before the first frame could match.
    Short episodes therefore scored 0.000 on a working decoder.
    """
    score = _score_over(monkeypatch, frame_count, sample_every=60)
    assert score > 0.0


def test_longer_segment_is_scored(monkeypatch):
    assert _score_over(monkeypatch, 120, sample_every=60) > 0.0


def test_no_frames_raises_instead_of_scoring_zero(monkeypatch):
    """
    An unscorable segment must raise. Returning 0.0 would remove episodes
    whose video was never actually inspected.
    """
    with pytest.raises(FrameDecodeError):
        _score_over(monkeypatch, 0)


def test_sampling_points_are_unchanged_for_longer_segments(monkeypatch):
    """
    The short-segment fix must not move the sampling points of segments that
    already reached them, or every existing score would shift. With
    sample_every=3 over 9 frames the measured frames stay 3, 6 and 9.
    """
    frames = [_sharp_frame() for _ in range(9)]
    position = {id(f): n for n, f in enumerate(frames, start=1)}
    seen = []
    monkeypatch.setattr(
        visual_module, "iterate_frames_in_range", lambda segment, **_: iter(frames)
    )
    monkeypatch.setattr(
        visual_module,
        "score_negative_visual_quality_opencv",
        lambda frame: seen.append(position[id(frame)]) or 0.0,
    )
    score_visual_clarity(
        _segment(), sts=None, acts=None, vlm=None, task=None, nom=None, sample_every=3
    )
    assert seen == [3, 6, 9]


def test_short_segment_measures_its_opening_frame(monkeypatch):
    """
    Below sample_every the opening frame is measured, and only that frame.
    """
    frames = [_sharp_frame() for _ in range(4)]
    position = {id(f): n for n, f in enumerate(frames, start=1)}
    seen = []
    monkeypatch.setattr(
        visual_module, "iterate_frames_in_range", lambda segment, **_: iter(frames)
    )
    monkeypatch.setattr(
        visual_module,
        "score_negative_visual_quality_opencv",
        lambda frame: seen.append(position[id(frame)]) or 0.0,
    )
    score_visual_clarity(
        _segment(), sts=None, acts=None, vlm=None, task=None, nom=None, sample_every=60
    )
    assert seen == [1]


def test_every_frame_is_sampled_when_interval_is_one(monkeypatch):
    calls = []
    monkeypatch.setattr(
        visual_module,
        "score_negative_visual_quality_opencv",
        lambda frame: calls.append(frame) or 0.25,
    )
    score = _score_over(monkeypatch, 5, sample_every=1)
    assert len(calls) == 5
    assert score == pytest.approx(0.75)
