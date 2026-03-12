import cv2
import subprocess
import numpy as np

# Camera Setup
width, height, fps = 1280, 720, 120
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# FFmpeg Command using Apple's Hardware Encoder (h264_videotoolbox)
command = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f"{width}x{height}",
    '-r', str(fps),
    '-i', '-',  # Input from pipe
    '-c:v', 'h264_videotoolbox',  # Apple Hardware Acceleration
    '-b:v', '10M',               # Bitrate
    'cicero_match_two.mp4'
]

proc = subprocess.Popen(command, stdin=subprocess.PIPE)

print("Recording... Press Ctrl+C to stop.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Write raw bytes directly to FFmpeg's stdin
        proc.stdin.write(frame.tobytes())
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    proc.stdin.close()
    proc.wait()
