import cv2
import time
import threading

class CameraStream:
    def __init__(self, src=1):
        self.cap = cv2.VideoCapture(src)
        # Force the camera to 60 FPS
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            print("update run")
            if ret:
                print("frame found")
                self.ret, self.frame = ret, frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- Testing Code ---
stream = CameraStream(src=1).start()
time.sleep(1.0) # Allow camera to warm up

for i in range(100):
    t0 = time.perf_counter()
    
    frame = stream.read() # This is now non-blocking
    
    duration = (time.perf_counter() - t0) * 1000
    print(f"Read time: {duration:.4f}ms")
    
    time.sleep(0.0166) # Your 40 FPS logic

stream.stop()