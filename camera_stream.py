import cv2
import threading
import time
import platform


class CameraStream:
    def __init__(self, src):
        # Use CAP_DSHOW if on Windows, if on Mac use AVFoundation, otherwise use default
        if platform.system() == "Windows":
            # print("Using DSHOW for Windows")
            # self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            print(
                "Temporarily using default backend for Windows (not DSHOW) due to issues.")
            self.cap = cv2.VideoCapture(src)
        elif platform.system() == "Darwin":
            print("Using AVFoundation for Mac")
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
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # 4. Set Frame Rate
        # self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FPS, 120)

        # 5. Buffer size (keep at 1 for low latency)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.frame_count = 0
        self.stopped = False

        # Print FPS, frame width, frame height of self.cap object
        print(f"Capture FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Capture Frame Width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(
            f"Capture Frame Height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

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


if __name__ == "__main__":
    # Initialize the stream
    # Change '0' to your specific camera index if needed

    camera_number = 0

    cam = CameraStream(src=camera_number).start()

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
