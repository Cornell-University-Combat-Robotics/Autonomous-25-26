from warp_main import get_warp_maps, warp_map
from main_helpers import key_frame, make_new_homography
import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# Add the project root to sys.path to allow imports
# This assumes segment.py is in the root directory 'Autonomous-25-26'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # --- Configuration ---
    # This script expects a YOLO segmentation model to be present at this path.
    MODEL_NAME = "NanoSegHueyPrince"
    MODEL_PATH = os.path.join("machine", "models", MODEL_NAME, "640", MODEL_NAME + ".mlpackage")
    # MODEL_PATH = os.path.join(
        # "machine", "models", MODEL_NAME, "640", MODEL_NAME + ".pt")

    VIDEO_PATH = os.path.join(
        "main_files", "test_videos", "huey_vs_prince.mp4")

    # --- Model Loading ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print(
            f"Please download a YOLO segmentation model (e.g., yolov8n-seg.pt) and place it in machine/models/ as {MODEL_NAME}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='segment')
    print("Model loaded.")

    # --- Video Input & Homography ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    print("\nPlease select a frame for homography by pressing '0'.")
    initial_frame = key_frame(cap, False, selection_scale=1.0)
    if initial_frame is None:
        print("No frame selected. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    print("\nSelect the 4 arena corners for the homography matrix.")
    # make_new_homography returns the warped frame and the matrix
    _, homography_matrix = make_new_homography(
        initial_frame, selection_scale=1.0)

    if homography_matrix is None:
        print("Homography matrix generation failed. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Get optimized warp maps and reset video
    map_x, map_y = get_warp_maps(homography_matrix)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\nStarting video playback with instance segmentation. Press 'q' to quit.")

    # --- Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # 1. Warp the frame
        warped_frame = warp_map(frame, map_x, map_y)

        # 2. Run segmentation (verbose=False prevents console spam)
        results = model.predict(warped_frame, verbose=True, task='segment')

        # 3. Plot results on the frame (.plot() returns a BGR numpy array with masks and boxes drawn)
        annotated_frame = results[0].plot()

        # 4. Display the result
        cv2.imshow("Instance Segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == "__main__":
    main()
