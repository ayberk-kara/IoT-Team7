import io
import requests
from threading import Condition, Thread
import time

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

server_ip = input("Input Server IP:")
address = f"http://{server_ip}:80"
desired_framerate = 1.0 / float(input("Input desired framerate:"))

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()
        self.latest_frame_time = time.time()

    def write(self, buf):
        with self.condition:
            current_time = time.time()
            if self.frame is None or current_time - self.latest_frame_time > desired_framerate:
                self.frame = buf
                self.latest_frame_time = current_time
                self.condition.notify_all()
                files = {'file': (f'{current_time}.jpg', buf, 'image/jpeg')}
                try:
                    requests.post(f"{address}/upload", files=files)  
                except requests.exceptions.RequestException as e:
                    print(f"Failed to upload frame: {e}")


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

while True:
    time.sleep(0.01)