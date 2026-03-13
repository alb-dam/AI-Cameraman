import numpy as np
import time
from cyndilib import Sender, VideoSendFrame, FourCC, AudioSendFrame

name = "Test NDI Audio Loop"
sender = Sender(name)
vf = VideoSendFrame()
vf.set_resolution(640, 480)
vf.set_frame_rate(30)
vf.set_fourcc(FourCC.BGRA)
sender.set_video_frame(vf)

af = AudioSendFrame()
sender.set_audio_frame(af)
sender.open()
time.sleep(1)

for i in range(10):
    audio_data = np.zeros((2, 1024), dtype=np.float32)
    try:
        sender.write_audio(audio_data)
        print(f"Audio frame {i} written.")
        time.sleep(0.03)
    except Exception as e:
        print(f"Error on frame {i}: {e}")

sender.close()
