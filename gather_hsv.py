import cv2
import numpy as np
import pandas as pd
import os

# --- Configuration ---
VIDEO_FILENAME = "huey_vs_prince.mp4"
FRAME_SKIP = 10  # Process every Nth frame
OUTPUT_FILENAME = "hsv_color_data.xlsx"

# --- Global Variables ---
selected_hsv_colors = []
current_frame = None
window_name = "Click colors, press '0' to skip frame, 'q' to quit"

def click_event(event, x, y, flags, params):
    """Mouse callback function to capture HSV color on click."""
    global selected_hsv_colors, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame is not None:
            # Get the BGR color of the clicked pixel
            bgr_color = current_frame[y, x]
            
            # Convert BGR to HSV
            # Note: cvtColor expects a 3D array, so we reshape
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Append to our list
            selected_hsv_colors.append(hsv_color)
            
            print(f"Color selected at ({x}, {y}): BGR={bgr_color}, HSV={hsv_color}. Total points: {len(selected_hsv_colors)}")
            
            # Draw a circle on the frame to give feedback
            # cv2.circle(current_frame, (x, y), 5, (0, 255, 0), 2)
            cv2.imshow(window_name, current_frame)

def main():
    """Main function to play video, gather data, and save to Excel."""
    global current_frame

    # Construct the path to the video
    video_path = os.path.join(os.getcwd(), "main_files", "test_videos", VIDEO_FILENAME)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    print("Starting video processing...")
    print("Instructions: Click on any number of points in the frame to select their color.")
    print("Press '0' to skip to the next sampled frame.")
    print("Press 'q' to finish and save the data.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        frame_count += 1

        # Only process every Nth frame
        if frame_count % FRAME_SKIP == 0:
            current_frame = frame.copy()  # Use a copy to draw on
            cv2.imshow(window_name, current_frame)
            
            # Wait for user input indefinitely
            key = cv2.waitKey(0) & 0xFF

            if key == ord('0'):
                print(f"Frame {frame_count} skipped.")
                continue
            elif key == ord('q'):
                print("Quitting video processing.")
                break
    
    # --- Cleanup and Save ---
    cap.release()
    cv2.destroyAllWindows()

    if not selected_hsv_colors:
        print("No colors were selected. Exiting without saving.")
        return

    print(f"\nTotal colors selected: {len(selected_hsv_colors)}")
    df = pd.DataFrame(np.array(selected_hsv_colors), columns=["Hue", "Saturation", "Value"])
    df.to_excel(OUTPUT_FILENAME, index=False)
    print(f"Successfully saved data to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()