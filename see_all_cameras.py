import cv2
from cv2_enumerate_cameras import enumerate_cameras

# Enumerate all available cameras
cameras = enumerate_cameras()

if cameras:
    print("Available cameras:")
    for camera_info in cameras:
        print(f"Index: {camera_info.index}, Name: {camera_info.name}, Path: {camera_info.path}")

    # Example: Open the first camera using its index
    if cameras[0].index is not None:
        cap = cv2.VideoCapture(cameras[0].index)
        if not cap.isOpened():
            print(f"Warning: unable to open camera at index {cameras[0].index}")
        else:
            print(f"Successfully opened camera: {cameras[0].name}")
            # You can now proceed with your video stream operations...
            cap.release()
else:
    print("No cameras found.")
