"""
Integration tests for issue #30, against a real ffmpeg and real video files.

The other two test modules stub ffmpeg out, which keeps them fast but means
they cannot prove the fix works against an actual decoder. These generate
video with ffmpeg and run the real iterate_frames_in_range over it.

Skipped automatically when ffmpeg is not on PATH.
"""

import shutil
import subprocess

import pytest

from score_lerobot_episodes.scores.visual import score_visual_clarity
from score_lerobot_episodes.util import FrameDecodeError, VideoSegment, iterate_frames_in_range

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH",
)


def _make_video(path, frames, fps=30, size="64x64"):
    """Render a real, decodable H.264 clip of exactly `frames` frames."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}",
         "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _segment(path, seconds=10.0):
    return VideoSegment(video_path=str(path), from_timestamp=0.0, to_timestamp=seconds)


@pytest.mark.parametrize("frames", [1, 10, 59])
def test_short_clip_scores_above_zero(tmp_path, frames):
    """
    The regression from #30: a clip shorter than sample_every decodes fine but
    used to sample nothing and score exactly 0.000, which the aggregate reads
    as unusable footage.
    """
    video = _make_video(tmp_path / f"short_{frames}.mp4", frames)

    score = score_visual_clarity(
        _segment(video), sts=None, acts=None, vlm=None, task=None, nom=None
    )

    assert score > 0.0, f"{frames}-frame clip scored {score}"


def test_long_clip_scores_above_zero(tmp_path):
    video = _make_video(tmp_path / "long.mp4", 90)
    score = score_visual_clarity(
        _segment(video), sts=None, acts=None, vlm=None, task=None, nom=None
    )
    assert score > 0.0


def test_real_decode_yields_expected_frame_count(tmp_path):
    video = _make_video(tmp_path / "count.mp4", 12)
    frames = list(iterate_frames_in_range(_segment(video)))
    assert len(frames) == 12
    assert frames[0].shape == (64, 64, 3)


def test_truncated_file_raises_rather_than_scoring(tmp_path):
    """
    A file ffmpeg cannot make sense of must raise, not come back as a score.
    Truncating to the first 400 bytes destroys the moov atom.
    """
    video = _make_video(tmp_path / "whole.mp4", 30)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(video.read_bytes()[:400])

    with pytest.raises(FrameDecodeError):
        score_visual_clarity(
            _segment(broken), sts=None, acts=None, vlm=None, task=None, nom=None
        )


def test_missing_file_raises_rather_than_scoring(tmp_path):
    with pytest.raises(FrameDecodeError):
        score_visual_clarity(
            _segment(tmp_path / "does_not_exist.mp4"),
            sts=None, acts=None, vlm=None, task=None, nom=None,
        )


def test_audio_only_file_raises(tmp_path):
    """No video stream to score. Previously surfaced as an opaque RuntimeError."""
    audio = tmp_path / "audio.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "aac", str(audio)],
        check=True, capture_output=True,
    )

    with pytest.raises(FrameDecodeError):
        list(iterate_frames_in_range(_segment(audio)))


@pytest.mark.xfail(
    reason="ffmpeg reads -t 0 as no duration limit, so a zero-length window "
           "decodes from the seek point to end of file and is scored as if it "
           "were the requested segment. Separate cause from this PR, left alone.",
)
def test_zero_length_window_is_not_scored_as_the_whole_file(tmp_path):
    """
    A zero-length window asks for no footage. Measured behaviour today:
    from=0.0 to=0.0 decodes all 30 frames and scores 1.0, and from=0.5 to=0.5
    decodes the trailing 15. Only a seek past the end returns nothing.
    """
    video = _make_video(tmp_path / "range.mp4", 30)
    segment = VideoSegment(video_path=str(video), from_timestamp=0.0, to_timestamp=0.0)

    frames = list(iterate_frames_in_range(segment))
    assert frames == [], f"zero-length window decoded {len(frames)} frames"
