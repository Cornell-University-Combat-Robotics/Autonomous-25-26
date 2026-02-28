import cv2
import threading
import time

class CameraStream:
    def __init__(self, src=0):
        # 1. Use DSHOW for lower latency image transfer
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # 3. Set MJPG Codec
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 2. Set Resolution to 720p (Crucial for 120fps on MK2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 4. Push for 120 FPS
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        
        # 5. Force hardware buffer to 1
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.stopped = False
        self.frame_count = 0
        
        # Use a Lock to prevent the main thread from reading 
        # while the camera thread is writing (prevents tearing)
        # self.read_lock = threading.Lock()

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
            # if self.cap.grab():
            #     # Only decode (retrieve) if the grab was successful
            #     ret, frame = self.cap.retrieve()
            #     self.frame_count = self.frame_count + 1

            #     # print(self.frame_count)

                
            #     # with self.read_lock:
            #     self.ret = ret
            #     self.frame = frame
            ret, frame = self.cap.read()
            self.ret = ret
            self.frame = frame

            self.frame_count = self.frame_count + 1
            print(self.frame_count)
            
            # REMOVED: time.sleep(). Let the OS scheduler handle the 
            # tight loop. This ensures we catch the USB packet the 
            # microsecond it arrives.

    def read(self):
        # with self.read_lock:
            # Return a copy if you find the main loop is 
            # modifying the frame, otherwise return direct for speed.
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        # Allow thread to finish its last loop
        time.sleep(0.1) 
        if self.cap.isOpened():
            self.cap.release()

    def isOpened(self):
        """Allows main.py to check if the camera is open just like a standard cv2 object."""
        return self.cap.isOpened() if self.cap else False

    def release(self):
        """Alias for stop() so main.py can use standard cv2 syntax."""
        self.stop()

    def frameCount(self):
        return self.frame_count

# --- MAIN METHOD IMPLEMENTATION ---

if __name__ == "__main__":
    # Initialize the stream
    # Change '0' to your specific camera index if needed
    cam = CameraStream(src=1).start()
    
    print("Camera Stream Started. Press 'q' to quit.")

    try:
        while True:
            # 1. Grab the most recent frame
            ret, frame = cam.read()

            # 2. Display if the frame is valid
            if ret and frame is not None:
                cv2.imshow("120FPS Camera Stream", frame)
            
            # 3. Use pollKey() for non-blocking input check
            # pollKey() returns -1 if no key is pressed
            key = cv2.pollKey() & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP: {e}")

    finally:
        # 4. Clean up resources
        print("Cleaning up...")
        cam.stop()
        cv2.destroyAllWindows()