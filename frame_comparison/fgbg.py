import cv2
import numpy as np
import os
import sys

# Add the parent directory to sys.path to allow imports from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_helpers import read_prev_homography
from warp_main import warp, get_warp_maps, warp_map

DO_WARP = True
USE_REFERENCE_FRAMES = False

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    video_path = os.path.join(base_dir, "main_files", "test_videos", "huey_vs_prince.mp4")
    ref_dir = os.path.join(current_dir, "reference_frames")
    homography_path = os.path.join(base_dir, "main_files", "homography_matrix.txt")

    # Load reference frames to create a static background model
    frames = []
    static_bg = None
    if USE_REFERENCE_FRAMES and os.path.exists(ref_dir):
        print("Loading reference frames for static background...")
        for i in range(1, 4):
            path = os.path.join(ref_dir, f"reference_frame_{i}.png")
            img = cv2.imread(path)
            if img is not None:
                frames.append(img)
        
        if frames:
            static_bg = np.median(np.array(frames), axis=0).astype(np.uint8)
            print("Static background computed.")

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

    # Initialize Background Subtractor (MOG2)
    # history: Length of the history.
    # varThreshold: Threshold on the squared Mahalanobis distance between the pixel and the model.
    # detectShadows: If True, the algorithm will detect shadows and mark them (usually as 127 in the mask).
    fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=40, detectShadows=True)
    
    print("Playing video with FGBG masking. Press 'q' to quit.")

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

        # Apply background subtraction to get the mask
        # Mask values: 0 (Background), 127 (Shadow), 255 (Foreground)
        fgmask = fgbg.apply(frame)

        # Threshold the mask to remove shadows (value 127), keeping only solid foreground (255)
        _, fgmask_clean = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # If we have a static background, compute difference and combine with MOG2 mask
        if static_bg is not None:
            diff = cv2.absdiff(frame, static_bg)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, static_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

            # Filter out large objects (Housebot) from the static mask
            contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_static_mask = np.zeros_like(static_mask)
            total_area = static_mask.shape[0] * static_mask.shape[1]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Keep if not too small (noise) and not too big (housebot > 5% of screen)
                if 500 < area < (0.03 * total_area):
                    cv2.drawContours(filtered_static_mask, [cnt], -1, 255, -1)

            # Combine masks: keeps object if it's moving (MOG2) OR if it's different from empty arena (Static)
            fgmask_clean = cv2.bitwise_or(fgmask_clean, filtered_static_mask)

        # Apply morphological opening to remove small noise (speckles)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_OPEN, kernel)

        # Attempt to split merged contours (e.g. two robots touching)
        # Typical robot area ~3000-5000. Merged ~9000.
        split_candidates, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in split_candidates:
            area = cv2.contourArea(cnt)
            if area > 6000:
                # Isolate this contour
                mask_roi = np.zeros_like(fgmask_clean)
                cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
                
                # Erode until split
                eroded = mask_roi.copy()
                split_kernel = np.ones((5, 5), np.uint8)
                was_split = False
                
                # Erode up to 15 times
                for _ in range(15):
                    eroded = cv2.erode(eroded, split_kernel, iterations=1)
                    # Check connectivity
                    num_labels, _ = cv2.connectedComponents(eroded)
                    if num_labels > 2: # 1 background + >1 objects
                        was_split = True
                        break
                    # Stop if it gets too small
                    if cv2.countNonZero(eroded) < area * 0.2:
                        break
                
                if was_split:
                    # Remove original large blob and add back the split parts
                    cv2.drawContours(fgmask_clean, [cnt], -1, 0, -1)
                    fgmask_clean = cv2.bitwise_or(fgmask_clean, eroded)

        # Apply clean mask to original frame
        masked_output = cv2.bitwise_and(frame, frame, mask=fgmask_clean)

        # Inspect contours every 30 frames
        if frame_count % 300 == 0:
            contours_inspect, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_inspect:
                total_area = fgmask_clean.shape[0] * fgmask_clean.shape[1]
                print(f"--- Frame {frame_count}: Inspecting contours ---")
                for i, cnt in enumerate(contours_inspect):
                    area = cv2.contourArea(cnt)
                    
                    if 500 < area < (0.03 * total_area):
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
        cv2.imshow("FGBG Mask (Raw)", fgmask)
        cv2.imshow("Foreground (Clean)", masked_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()