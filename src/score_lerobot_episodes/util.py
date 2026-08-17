from dataclasses import dataclass
import threading
import ffmpeg
import numpy as np
from score_lerobot_episodes.vlm import VLMInterface

class FrameDecodeError(RuntimeError):
    """
    Raised when frames cannot be decoded from a video segment.

    Scores feed a keep/discard decision, so a segment that yields no frames
    must fail loudly rather than resolve to a score.
    """

@dataclass
class VideoSegment:
    video_path: str
    from_timestamp: float
    to_timestamp: float

def iterate_frames_in_range(video_segment: VideoSegment, output_width=-1):
    """
    Iterates over frames within a specified time range using ffmpeg-python.
    """
    video_path = video_segment.video_path
    start_time_seconds = video_segment.from_timestamp
    duration_seconds = video_segment.to_timestamp - video_segment.from_timestamp

    try:
        # Probe the video to get properties
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream is None:
            raise FrameDecodeError(f"No video stream found in {video_path}")
        width = video_stream['width']
        height = video_stream['height']

        # Set up the ffmpeg input and output streams
        # -ss (seek) before -i is for fast seeking to a keyframe near the start time
        # -t (duration) specifies how long to process from the seek point
        process = (
            ffmpeg
            .input(video_path, ss=start_time_seconds, t=duration_seconds)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=999999) # Set vframes high enough to cover the duration
            .global_args("-nostdin")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Drain stderr on a thread. ffmpeg blocks once the stderr pipe buffer
        # fills, so reading it only after the frame loop would deadlock on a
        # decoder that logs a line per frame.
        stderr_chunks = []
        stderr_reader = threading.Thread(
            target=lambda pipe, sink: sink.append(pipe.read()),
            args=(process.stderr, stderr_chunks),
            daemon=True,
        )
        stderr_reader.start()

        try:
            frame_size = width * height * 3  # 3 bytes for rgb24
            while True:
                # Read frame from the stdout pipe
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    break
                if len(in_bytes) != frame_size:
                    # Otherwise reshape fails with "cannot reshape array of
                    # size N", which says nothing about which video or why.
                    raise FrameDecodeError(
                        f"Truncated frame from {video_path}: got "
                        f"{len(in_bytes)} bytes, expected {frame_size}")

                # Convert the raw bytes to a numpy array (frame)
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

                yield frame

            # A non-zero exit means ffmpeg gave up part way through, or never
            # decoded anything at all. Without this check the caller cannot
            # tell that apart from an empty time range.
            returncode = process.wait()
            stderr_reader.join(timeout=5)
            if returncode != 0:
                stderr = b''.join(stderr_chunks).decode('utf8', errors='replace')
                raise FrameDecodeError(
                    f"ffmpeg exited {returncode} while decoding {video_path}: "
                    f"{stderr.strip()}")
        finally:
            # A caller that stops early, or whose own per-frame work raises,
            # abandons this generator with ffmpeg still running and its pipes
            # still open. Reap it, or a long batch accumulates one process and
            # one reader thread per abandoned segment.
            if process.poll() is None:
                process.kill()
                process.wait()
            stderr_reader.join(timeout=5)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf8', errors='replace') if e.stderr else ''
        raise FrameDecodeError(
            f"ffprobe failed for {video_path}: {stderr.strip()}") from e

