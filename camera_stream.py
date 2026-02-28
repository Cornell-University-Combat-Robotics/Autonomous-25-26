import cv2
import threading
import time

class CameraStream:
    def __init__(self, src=0):
        # 1. Switch to MSMF for modern Windows/Elgato support
        self.cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
        
        # 2. Set Resolution to 720p (Crucial for 120fps on MK2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 3. Set MJPG Codec
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 4. Push for 120 FPS
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        
        # 5. Force hardware buffer to 1
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.stopped = False
        
        # Use a Lock to prevent the main thread from reading 
        # while the camera thread is writing (prevents tearing)
        self.read_lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break

            # Use grab() to clear the buffer as fast as possible
            # This is non-blocking and extremely fast
            if self.cap.grab():
                # Only decode (retrieve) if the grab was successful
                ret, frame = self.cap.retrieve()
                
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            
            # REMOVED: time.sleep(). Let the OS scheduler handle the 
            # tight loop. This ensures we catch the USB packet the 
            # microsecond it arrives.

    def read(self):
        with self.read_lock:
            # Return a copy if you find the main loop is 
            # modifying the frame, otherwise return direct for speed.
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        # Allow thread to finish its last loop
        time.sleep(0.1) 
        if self.cap.isOpened():
            self.cap.release()