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
    ref_dir = os.path.join(current_dir, "reference_frames")
    video_path = os.path.join(base_dir, "main_files", "test_videos", "huey_vs_prince.mp4")
    homography_path = os.path.join(base_dir, "main_files", "homography_matrix.txt")

    # Load reference frames
    frames = []
    print("Loading reference frames...")
    for i in range(1, 4):
        path = os.path.join(ref_dir, f"reference_frame_{i}.png")
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read {path}. Please run frame_comparison.py first to select frames.")
            return
        frames.append(img)

    # Create background model using Median filtering
    # We stack the frames to shape (3, Height, Width, 3)
    # Then take the median along axis 0 (the frame axis)
    # This selects the pixel value that appears in the middle of the sorted list for each position
    print("Computing background model (Median)...")
    frame_stack = np.array(frames)
    background = np.median(frame_stack, axis=0).astype(np.uint8)
    background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    cv2.imshow("Computed Background (Clean Plate)", background)
    print("Background computed. Press any key to start video...")
    cv2.waitKey(0)
    cv2.destroyWindow("Computed Background (Clean Plate)")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    homography_matrix = None
    map_x, map_y = None, None
    if DO_WARP:
        # Read the first frame to load the homography matrix
        ret, first_frame = cap.read()
        if ret:
            # read_prev_homography reads the text file and returns the matrix
            _, homography_matrix = read_prev_homography(first_frame, homography_path)
            if homography_matrix is not None:
                map_x, map_y = get_warp_maps(homography_matrix)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video

    print("Playing video with background subtraction. Press 'q' to quit.")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Warp the frame before processing
        if DO_WARP and map_x is not None and map_y is not None:
            frame = warp_map(frame, map_x, map_y)
        elif DO_WARP and homography_matrix is not None:
            frame = warp(frame, homography_matrix)

        # Convert frame to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Split channels for background and current frame
        h_bg, s_bg, v_bg = cv2.split(background_hsv)
        h_frame, s_frame, v_frame = cv2.split(frame_hsv)

        # Calculate Hue Difference (handling 180 wrap-around)
        h_diff_int = np.abs(h_frame.astype(np.int16) - h_bg.astype(np.int16))
        h_diff = np.minimum(h_diff_int, 180 - h_diff_int).astype(np.uint8)

        # Calculate S and V differences
        s_diff = cv2.absdiff(s_frame, s_bg)
        v_diff = cv2.absdiff(v_frame, v_bg)

        # Thresholds: Strict on Hue (20), Loose on Sat/Val (80)
        _, mask_h = cv2.threshold(h_diff, 10, 255, cv2.THRESH_BINARY)
        _, mask_s = cv2.threshold(s_diff, 100, 255, cv2.THRESH_BINARY)
        _, mask_v = cv2.threshold(v_diff, 100, 255, cv2.THRESH_BINARY)

        # Combine masks
        mask = cv2.bitwise_or(mask_h, mask_s)
        mask = cv2.bitwise_or(mask, mask_v)

        # Apply morphological operations to remove noise and close holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original frame to show only changed pixels
        masked_output = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours for filtering and inspection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create mask for contours > 500
        mask_500 = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(mask_500, [cnt], -1, 255, -1)
        output_500 = cv2.bitwise_and(frame, frame, mask=mask_500)

        # Inspect contours every 60 frames
        if frame_count % 60 == 0 and False:
            if contours:
                print(f"--- Frame {frame_count}: Inspecting contours > 100 ---")
                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area > 300:
                        print(f"Contour {i+1}: Area = {area}")
                        inspect_img = frame.copy()
                        cv2.drawContours(inspect_img, [cnt], -1, (0, 255, 0), 2)
                        cv2.putText(inspect_img, f"Area: {area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Contour Inspection", inspect_img)
                        cv2.waitKey(0)
                try:
                    cv2.destroyWindow("Contour Inspection")
                except:
                    pass

        cv2.imshow("Original", frame)
        cv2.imshow("Masked Foreground", masked_output)
        cv2.imshow("Contours Over 500", output_500)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()