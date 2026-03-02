import cv2
import threading
import time
import platform

class CameraStream:
    def __init__(self, src):
        # Use CAP_DSHOW if on Windows, if on Mac use AVFoundation, otherwise use default
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        elif platform.system() == "Darwin":
            self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(src)
        
        # 2. Set Codec FIRST (Essential for Elgato bandwidth)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Elgato FaceCam Mk2 can capture at 1080p 60fps or 720p 120fps, among others
        
        # 3. Set Resolution
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 4. Set Frame Rate
        # self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        
        # 5. Buffer size (keep at 1 for low latency)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
                
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
                self.frame_count = self.frame_count + 1

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