import cv2
import threading
import time

class CameraStream:
    def __init__(self, src):
        # 1. Use the 'sum' trick for Windows DirectShow
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # 2. Set Codec FIRST (Essential for Elgato bandwidth)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
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
                
                
            # print("try GRAB")
                
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
                self.frame_count = self.frame_count + 1
                # print("Time between grabs: " + str(time.time() - last_success))
                last_success = time.time()
            else:
                # If the camera hiccups, don't kill the thread immediately
                # time.sleep(0.001) 
                continue
            
            # time.sleep(0.003)

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