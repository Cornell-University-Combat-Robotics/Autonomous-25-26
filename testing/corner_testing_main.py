
import os
import sys
import cv2
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from corner_testing_helpers import draw_orientation_arrow, get_darkness_score, angle_difference, draw_manual_arrow
from corner_detection.corner_detection import RobotCornerDetection
from main_helpers import (
    make_new_colors,
    initialize_quantization,
    quantize
)

#Settings
SAMPLE = True
NONE_SCORE = 45 #How "bad" is it to get no orientation

# Change this to your folder path
FOLDER_PATH = "testing/testing_data/huey_unquantized"
LABELED_DATA_PATH = os.path.join(FOLDER_PATH, "angles_output.csv")
folder = os.getcwd() + "/testing"

#Set up angle lookup
df = pd.read_csv(LABELED_DATA_PATH)
angle_lookup = dict(zip(df["filename"], df["angle"]))

# Valid image file extensions
IMAGE_EXTENSIONS = (".png")

first_heuy_path = os.path.join(FOLDER_PATH, "1061.png")
first_huey = cv2.imread(first_heuy_path)
selected_colors = make_new_colors(folder + "/selected_colors.txt", first_huey)
print("COLORS:")
print(selected_colors)

corner_detection = RobotCornerDetection(selected_colors, False, False)

def test_detect_corners(quant_settings=False):
    total_frames = 0
    frames_with_orientation = 0

    total_theta = 0

    total_score = 0 #If orientation, theta. If none, 90

    initialize_quantization() #PASS IN SETTINGS!!!

    for filename in os.listdir(FOLDER_PATH):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_path = os.path.join(FOLDER_PATH, filename)
            
            print(f"Processing: {image_path}")
            
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load: {image_path}")
                continue
            
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
            
            quantized_bots = quantize(formated_image, selected_colors, show=False, is_flipped=False, settings=quant_settings)
            quantized_img = quantized_bots['bots'][0]['img']

            corner_detection.set_bots(quantized_bots)
            detected_bots_with_data = corner_detection.corner_detection_main()
            # print(detected_bots_with_data)
            
            bbox = detected_bots_with_data['huey']['bbox']
            x, y, w, h = bbox
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            
            true_angle = angle_lookup.get(filename)

            total_frames += 1
            if detected_bots_with_data['huey']['orientation'] != None:
                frames_with_orientation += 1
                if true_angle:
                    current_angle_difference = angle_difference(true_angle, detected_bots_with_data['huey']['orientation'])
                    total_theta += current_angle_difference
                    total_score += current_angle_difference
            else:
                total_score += NONE_SCORE

            if SAMPLE:
                print(f"Correct Angle: {angle_lookup.get(filename)}")
                print(f"Calculated Angle: {detected_bots_with_data['huey']['orientation']}")
                if detected_bots_with_data['huey']['orientation'] != None:
                    draw_orientation_arrow(quantized_img, detected_bots_with_data)
                    draw_manual_arrow(quantized_img, cx, cy, true_angle)

                    if true_angle:
                        print(f"Angle difference: {current_angle_difference}")
                
                cv2.imshow("Image Viewer", quantized_img)
                print(f"Showing: {filename} (quantized)")

                key = cv2.waitKey(0)  # Wait for key press
                if key == ord('n'):   # Press 'n' to move to next setting
                    print("--------------------------------")
                    print(f"Settings: {quant_settings}")
                    print(f"Frames with orientation: {frames_with_orientation} / {total_frames}")
                    print(f"Average Theta: {total_theta/max(frames_with_orientation, 1)}")
                    print(f"Score: {total_score/total_frames}")
                    print("--------------------------------")
                    return (total_score/total_frames)
    if SAMPLE:
        cv2.destroyAllWindows()

    return (total_score/total_frames)

def test_detect_corners_black_box(threshold, L_weight, RG_weight, BY_weight):
    return test_detect_corners(quant_settings={
        "threshold": threshold,
        "quantization_weights": [L_weight, RG_weight, BY_weight]
    })

#MAIN:
print("PRESS 0 TO SWITCH IMAGES AND N TO ITERATE QUANTIZATION SETTINGS")

orientation_scores = {}

for i in range(20, 45, 3):
    orientation_scores[i] = test_detect_corners(quant_settings={
        "threshold": i,
        "quantization_weights": [0.2, 0.4, 0.4]
    })

print(str(orientation_scores))