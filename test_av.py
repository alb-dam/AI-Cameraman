import av
import numpy as np
import subprocess
import os

if not os.path.exists("test_vid.mp4"):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=640x480:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "test_vid.mp4"
    ], capture_output=True)

container = av.open("test_vid.mp4")
video_stream = container.streams.video[0]
audio_stream = container.streams.audio[0] if container.streams.audio else None

print("Audio Stream:", audio_stream)
for frame in container.decode(video_stream, audio_stream):
    if isinstance(frame, av.AudioFrame):
        print("Audio default:", frame.to_ndarray().shape, frame.to_ndarray().dtype)
        # Using fltp, we should get (channels, samples) shape
        arr = frame.to_ndarray(format="fltp")
        print("Audio fltp:", arr.shape, arr.dtype)
        break
for frame in container.decode(video_stream, audio_stream):
    if isinstance(frame, av.VideoFrame):
        print("Video:", frame.to_ndarray(format="bgr24").shape)
        break
