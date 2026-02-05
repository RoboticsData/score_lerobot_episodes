from dataclasses import dataclass
import ffmpeg
import numpy as np
from score_lerobot_episodes.vlm import VLMInterface

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
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
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

        frame_size = width * height * 3  # 3 bytes for rgb24
        while True:
            # Read frame from the stdout pipe
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            
            # Convert the raw bytes to a numpy array (frame)
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            
            # Process the frame (e.g., display, analyze, save)
            # print(f"Processing frame of shape: {frame.shape}") 
            yield frame

        process.wait()
    except ffmpeg.Error as e:
        print('FFmpeg Error:', e.stderr.decode('utf8'))

