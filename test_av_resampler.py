import av

container = av.open("test_vid.mp4")
audio_stream = container.streams.audio[0]

# We want float32 planar output (fltp) for cyndilib NDI.
resampler = av.AudioResampler(format='fltp', layout='stereo')

for frame in container.decode(audio_stream):
    frame.pts = None
    resampled_frames = resampler.resample(frame)
    for r_frame in resampled_frames:
        arr = r_frame.to_ndarray()
        print(f"Resampled Audio: shape={arr.shape}, dtype={arr.dtype}")
    break
