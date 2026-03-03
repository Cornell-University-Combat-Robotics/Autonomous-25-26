
import os
import sys
import cv2
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from corner_testing_helpers import draw_orientation_arrow, get_darkness_score
from corner_detection.corner_detection import RobotCornerDetection
from main_helpers import (
    display_angles,
    first_run,
    get_motor_groups,
    get_predictor,
    key_frame,
    make_new_colors,
    make_new_homography,
    read_prev_colors,
    read_prev_homography,
    initialize_quantization,
    quantize
)

#Settings
SAMPLE = True

# Change this to your folder path
FOLDER_PATH = "testing/testing_data/huey_unquantized"
LABELED_DATA_PATH = os.path.join(FOLDER_PATH, "angles_output.csv")
folder = os.getcwd() + "/testing"

#Set up angle lookup
df = pd.read_csv(LABELED_DATA_PATH)
angle_lookup = dict(zip(df["filename"], df["angle"]))

# Valid image file extensions
IMAGE_EXTENSIONS = (".png")

first_heuy_path = os.path.join(FOLDER_PATH, "1.png")
first_huey = cv2.imread(first_heuy_path)
selected_colors = make_new_colors(folder + "/selected_colors.txt", first_huey)
print("COLORS:")
print(selected_colors)

corner_detection = RobotCornerDetection(selected_colors, False, False)

def detect_corners(quant_settings=False):
    total_frames = 0
    frames_with_orientation = 0

    for filename in os.listdir(FOLDER_PATH):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_path = os.path.join(FOLDER_PATH, filename)
            
            print(f"Processing: {image_path}")
            
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load: {image_path}")
                continue
            
            # get_overquantize_score(image, selected_colors)

            height, width = image.shape[:2]

            # Fake bounding box, the whole image
            bbox = (0, 0, width, height)
            formated_image = {
                "bots": [
                    {
                        'img': image,
                        'bbox': bbox
                    }
                ]
            }
            
            initialize_quantization()
            quantized_bots = quantize(formated_image, selected_colors, show=False, is_flipped=False, settings=quant_settings)
            quantized_img = quantized_bots['bots'][0]['img']

            corner_detection.set_bots(quantized_bots)
            detected_bots_with_data = corner_detection.corner_detection_main()

            total_frames += 1
            if detected_bots_with_data['huey']['orientation'] != None:
                frames_with_orientation += 1

            # print(detected_bots_with_data)

            print(f"Correct Angle: {angle_lookup.get(filename)}")
            print(f"Calculated Angle: {detected_bots_with_data['huey']['orientation']}")


            if SAMPLE:
                if detected_bots_with_data['huey']['orientation'] != None:
                    draw_orientation_arrow(quantized_img, detected_bots_with_data)
                cv2.imshow("Image Viewer", quantized_img)
                print(f"Showing: {filename} (quantized)")

                key = cv2.waitKey(0)  # Wait for key press
                if key == ord('n'):   # Press 'q' to quit early
                    return frames_with_orientation, total_frames
    if SAMPLE:
        cv2.destroyAllWindows()

    return frames_with_orientation, total_frames

print("PRESS 0 TO SWITCH IMAGES AND N TO ITERATE QUANTIZATION SETTINGS")

orientation_scores = {}

for i in range(20, 80, 10):


    frames_with_orientation, total_frames = detect_corners(quant_settings={
        "threshold": i
    })

    # orientation_scores[i]['Orientation (captured, total)'] = (frames_with_orientation, total_frames)
    # print(f"Threshold = {i}% Frames with orientation {frames_with_orientation}/{total_frames} => {(frames_with_orientation/total_frames)*100:.002f}%")

print(str(orientation_scores))