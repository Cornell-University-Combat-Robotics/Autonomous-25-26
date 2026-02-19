import cv2
import threading
import time

class CameraStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        
        # Performance tuning for Elgato MK2
        # MJPG is often faster for high-res 60fps
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.ret, self.frame = self.cap.read()
        self.frame_count = 0
        self.stopped = False

    def start(self):
        # Using daemon=True so the thread stops when main.py exits
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        last_success = time.time()
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                
                
            # print("try GRAB")
                
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
                self.frame_count = self.frame_count + 1
                # print("Time between grabs: " + str(time.time() - last_success))
                last_success = time.time()
            else:
                # If the camera hiccups, don't kill the thread immediately
                time.sleep(0.001) 
            
            time.sleep(0.003)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()
    
    def frameCount(self):
        return self.frame_count