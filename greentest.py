import cv2
import numpy as np
from camera_stream import CameraStream
import time

def main():
    # Initialize camera stream using source 1 (as used in main.py)
    cam = CameraStream(src=0).start()

    print("Green detection started. Press 'q' to quit.")

    last_frame = 0

    try:
        while True:
            # Capture frame from camera
            ret, frame = cam.read()

            if not ret or frame is None:
                continue

            # # Convert the frame to HSV color space for more robust color detection
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # # Define the HSV range for green
            # # Note: These values might need tuning based on your specific lighting conditions
            # lower_green = np.array([35, 50, 50])
            # upper_green = np.array([85, 255, 255])

            # # Create a mask that identifies green pixels
            # mask = cv2.inRange(hsv, lower_green, upper_green)

            # # Calculate the percentage of green pixels in the image
            # green_pixels = cv2.countNonZero(mask)
            # total_pixels = frame.shape[0] * frame.shape[1]
            # green_percentage = (green_pixels / total_pixels) * 100

            # # Print "GREEN" if more than 50% of the image is green
            # if green_percentage > 50:
            #     print("GREEN at " + str(time.perf_counter()))
            if cam.frameCount() > last_frame:
                last_frame = cam.frameCount()
                print(f"Frame count: {cam.frameCount()} at {time.perf_counter()%10:.3f}")
            # print(str(time.perf_counter()%10)
            # print(time.perf_counter() % 10)

            # Display the camera feed
            # cv2.imshow("Green Detection Feed", frame)

            # Exit loop if 'q' is pressed
            # if cv2.pollKey() & 0xFF == ord('q'):
            #     break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
