"""
Tests for issue #30: a failed ffmpeg run must not look like an empty segment.

iterate_frames_in_range previously discarded the ffmpeg return code and
printed probe failures, so both cases yielded no frames and were
indistinguishable from a legitimately empty time range.

ffmpeg is stubbed out here, so these need no ffmpeg binary and no video files.
"""

import gc

import numpy as np
import pytest

from score_lerobot_episodes import util as util_module
from score_lerobot_episodes.util import FrameDecodeError, VideoSegment, iterate_frames_in_range


class _FakePipe:
    def __init__(self, chunks):
        self._data = b"".join(chunks)
        self._pos = 0

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


class _FakeProcess:
    def __init__(self, frames=(), stderr=b"", returncode=0):
        self.stdout = _FakePipe(frames)
        self.stderr = _FakePipe([stderr])
        self._returncode = returncode
        self._reaped = False
        self.killed = False

    def wait(self):
        self._reaped = True
        return self._returncode

    def poll(self):
        return self._returncode if self._reaped else None

    def kill(self):
        self.killed = True


def _stub_ffmpeg(monkeypatch, process, *, width=2, height=2, probe_error=None):
    """Replace the ffmpeg module surface that iterate_frames_in_range touches."""

    class _Stub:
        Error = util_module.ffmpeg.Error

        @staticmethod
        def probe(path):
            if probe_error is not None:
                raise probe_error
            return {"streams": [{"codec_type": "video", "width": width, "height": height}]}

        @staticmethod
        def input(*a, **k):
            return _Stub()

        def output(self, *a, **k):
            return self

        def global_args(self, *a, **k):
            return self

        def run_async(self, *a, **k):
            return process

    monkeypatch.setattr(util_module, "ffmpeg", _Stub)


def _segment():
    return VideoSegment(video_path="stub.mp4", from_timestamp=0.0, to_timestamp=1.0)


def _rgb_frame(width=2, height=2, value=7):
    return bytes([value]) * (width * height * 3)


def test_nonzero_exit_raises_with_ffmpeg_stderr(monkeypatch):
    """
    The decode failure that started #30: ffmpeg exits non-zero having produced
    no frames. Previously the return code was dropped and the caller saw an
    empty iterator.
    """
    process = _FakeProcess(
        frames=[],
        stderr=b"Your platform doesn't support hardware accelerated AV1 decoding.\n",
        returncode=1,
    )
    _stub_ffmpeg(monkeypatch, process)

    with pytest.raises(FrameDecodeError) as excinfo:
        list(iterate_frames_in_range(_segment()))

    assert "exited 1" in str(excinfo.value)
    # The reason ffmpeg gave must survive into the error, not just the console.
    assert "AV1" in str(excinfo.value)


def test_nonzero_exit_after_partial_output_still_raises(monkeypatch):
    """A run that dies part way through is a failure, not a short segment."""
    process = _FakeProcess(
        frames=[_rgb_frame(), _rgb_frame()], stderr=b"boom", returncode=254
    )
    _stub_ffmpeg(monkeypatch, process)

    produced = []
    with pytest.raises(FrameDecodeError):
        for frame in iterate_frames_in_range(_segment()):
            produced.append(frame)

    assert len(produced) == 2


def test_clean_exit_yields_frames_and_does_not_raise(monkeypatch):
    process = _FakeProcess(frames=[_rgb_frame(), _rgb_frame()], returncode=0)
    _stub_ffmpeg(monkeypatch, process)

    frames = list(iterate_frames_in_range(_segment()))

    assert len(frames) == 2
    assert frames[0].shape == (2, 2, 3)
    assert frames[0].dtype == np.uint8


def test_probe_failure_raises_instead_of_printing(monkeypatch):
    """
    A probe failure used to be printed, after which the generator returned
    normally and the caller scored the segment as if it were empty.
    """
    error = util_module.ffmpeg.Error("ffprobe", b"", b"moov atom not found")
    _stub_ffmpeg(monkeypatch, _FakeProcess(), probe_error=error)

    with pytest.raises(FrameDecodeError) as excinfo:
        list(iterate_frames_in_range(_segment()))

    assert "ffprobe failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, util_module.ffmpeg.Error)


def test_abandoning_the_generator_kills_ffmpeg(monkeypatch):
    """
    A caller that stops early, or whose per-frame work raises, must not leave
    ffmpeg running. Otherwise a long batch accumulates one process and one
    stderr reader thread per abandoned segment.
    """
    process = _FakeProcess(frames=[_rgb_frame() for _ in range(5)], returncode=0)
    _stub_ffmpeg(monkeypatch, process)

    for frame in iterate_frames_in_range(_segment()):
        break                      # take one frame and walk away

    gc.collect()                   # force the generator to be finalised
    assert process.killed


def test_consumer_exception_kills_ffmpeg(monkeypatch):
    process = _FakeProcess(frames=[_rgb_frame() for _ in range(5)], returncode=0)
    _stub_ffmpeg(monkeypatch, process)

    with pytest.raises(ValueError):
        for frame in iterate_frames_in_range(_segment()):
            raise ValueError("per-frame work failed")

    gc.collect()
    assert process.killed


def test_fully_consumed_generator_does_not_kill_ffmpeg(monkeypatch):
    """Clean runs must be waited on, not killed."""
    process = _FakeProcess(frames=[_rgb_frame(), _rgb_frame()], returncode=0)
    _stub_ffmpeg(monkeypatch, process)

    list(iterate_frames_in_range(_segment()))

    assert not process.killed


def test_missing_video_stream_raises(monkeypatch):
    """
    An audio-only file has no video stream. next() with no default would raise
    StopIteration, which Python turns into RuntimeError inside a generator.
    """
    class _AudioOnly:
        Error = util_module.ffmpeg.Error

        @staticmethod
        def probe(path):
            return {"streams": [{"codec_type": "audio"}]}

    monkeypatch.setattr(util_module, "ffmpeg", _AudioOnly)

    with pytest.raises(FrameDecodeError) as excinfo:
        list(iterate_frames_in_range(_segment()))

    assert "No video stream" in str(excinfo.value)
