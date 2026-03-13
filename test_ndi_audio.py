import numpy as np
import time
from cyndilib import Sender, VideoSendFrame, AudioSendFrame, FourCC

name = "Test NDI Audio"
sender = Sender(name)

vf = VideoSendFrame()
vf.set_resolution(640, 480)
vf.set_frame_rate(30)
vf.set_fourcc(FourCC.BGRA)
sender.set_video_frame(vf)

# Attempt to configure audio frame
af = AudioSendFrame()
# set properties? let's see if there are setters
try:
    af.sample_rate = 48000
    af.num_channels = 2
except Exception as e:
    print("Cannot set directly:", e)
    
sender.set_audio_frame(af)
sender.open()

print("Sender opened.")
time.sleep(1)

# try sending audio
audio_data = np.zeros((2, 1024), dtype=np.float32)

try:
    print("Writing audio...")
    sender.write_audio(audio_data)
    print("Audio written.")
except Exception as e:
    print("Error writing audio:", e)

sender.close()
