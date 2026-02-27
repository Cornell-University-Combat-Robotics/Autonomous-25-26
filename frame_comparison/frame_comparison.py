import cv2
import numpy as np
import os
import sys

# Add the parent directory to sys.path to allow imports from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_helpers import key_frame, make_new_homography
from warp_main import warp

DO_WARP = True

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(base_dir, "main_files", "test_videos", "huey_vs_prince.mp4")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_frames")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    homography_matrix = None
    if DO_WARP:
        print("Warp enabled. Reading first frame for homography selection...")
        ret, first_frame = cap.read()
        if ret:
            # Generates matrix, saves to file, and returns warped frame + matrix
            _, homography_matrix = make_new_homography(first_frame)
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Video loaded. Press '0' to select a frame, 'q' to quit.")

    # Select 3 frames
    for i in range(1, 4):
        print(f"Selecting frame {i} of 3...")
        
        # Use key_frame from main_helpers to handle playback and selection
        # key_frame returns the frame when '0' is pressed, or None if 'q' or error
        selected_frame = key_frame(cap, False)

        if selected_frame is None:
            print("Selection cancelled or video ended.")
            break

        # Warp the frame if enabled
        if DO_WARP and homography_matrix is not None:
            selected_frame = warp(selected_frame, homography_matrix)

        # Save the frame
        output_path = os.path.join(output_dir, f"reference_frame_{i}.png")
        cv2.imwrite(output_path, selected_frame)
        print(f"Saved: {output_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
