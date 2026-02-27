import cv2
import numpy as np
import os
import sys

# Add the parent directory to sys.path to allow imports from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_helpers import read_prev_homography
from warp_main import warp, get_warp_maps, warp_map

DO_WARP = True

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    video_path = os.path.join(base_dir, "main_files", "test_videos", "huey_vs_prince.mp4")
    homography_path = os.path.join(base_dir, "main_files", "homography_matrix.txt")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Warping setup
    homography_matrix = None
    map_x, map_y = None, None
    if DO_WARP:
        ret, first_frame = cap.read()
        if ret:
            _, homography_matrix = read_prev_homography(first_frame, homography_path)
            if homography_matrix is not None:
                map_x, map_y = get_warp_maps(homography_matrix)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video

    # Read first frame
    ret, frame1 = cap.read()
    if not ret:
        return

    # Warp first frame if enabled
    if DO_WARP and map_x is not None and map_y is not None:
        frame1 = warp_map(frame1, map_x, map_y)
    elif DO_WARP and homography_matrix is not None:
        frame1 = warp(frame1, homography_matrix)

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    print("Playing video with motion tracking. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret: break
        
        # Warp current frame
        if DO_WARP and map_x is not None and map_y is not None:
            frame2 = warp_map(frame2, map_x, map_y)
        elif DO_WARP and homography_matrix is not None:
            frame2 = warp(frame2, homography_matrix)

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 1. Get raw motion
        diff = cv2.absdiff(prvs, next_frame)
        _, raw_motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 2. Solidify dark objects and remove tiny noise
        kernel = np.ones((15, 15), np.uint8)
        solid_motion = cv2.morphologyEx(raw_motion_mask, cv2.MORPH_CLOSE, kernel)
        solid_motion = cv2.morphologyEx(solid_motion, cv2.MORPH_OPEN, kernel)

        # 3. Find Contours
        contours, _ = cv2.findContours(solid_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original frame to verify
        display_frame = frame2.copy()
        cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
        
        cv2.imshow('Tracked Contours', display_frame)

        prvs = next_frame

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()