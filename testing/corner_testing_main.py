
import os
import sys
import cv2
import time
import math
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
DISPLAY_IMAGES = True           # Displays each image in window
NO_ORIENTATION_SCORE = 25       # Angle that is equivelently bad to no orientation
DATA_SET_NAME = "prince_full"   # Name of your dataset folder in "testing_data"
COLOR_SELECT_IMG = "900.png"    # This should be the name of your image you want to do color selection on

# Folder paths
ALL_TRAINING_DATA_PATH = "testing_data/" 
FOLDER_PATH = os.path.join(ALL_TRAINING_DATA_PATH, DATA_SET_NAME)
FIRST_HUEY_PATH = os.path.join(FOLDER_PATH, COLOR_SELECT_IMG)

LABELED_DATA_PATH = os.path.join(FOLDER_PATH, "angles_output.csv")
TESTING_DIR = os.getcwd()

#Set up angle lookup
df = pd.read_csv(LABELED_DATA_PATH)
angle_lookup = dict(zip(df["filename"], df["angle"]))

# Valid image file extensions
IMAGE_EXTENSIONS = (".png")

first_huey = cv2.imread(FIRST_HUEY_PATH)
selected_colors = make_new_colors(TESTING_DIR + "/selected_colors.txt", first_huey)
print("Selected Colors:")
print(selected_colors)

corner_detection = RobotCornerDetection(selected_colors, False, False)

def test_detect_corners(threshold, L_weight, RG_weight, BY_weight, area_threshold, DISPLAY_IMAGES=DISPLAY_IMAGES, NO_ORIENTATION_SCORE=NO_ORIENTATION_SCORE):

    #Format quantization settings
    quant_settings={
        "threshold": threshold,
        "area_theshold": area_threshold,
        "quantization_weights": [L_weight, RG_weight, BY_weight]
    }

    total_frames = 0
    frames_with_orientation = 0

    total_theta = 0 #Keeps track of true sum of angle differences

    total_score = 0 #If orientation, theta. If none, add selected score

    initialize_quantization()

    #Loop through all images in dataset folder
    for filename in os.listdir(FOLDER_PATH): 
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_path = os.path.join(FOLDER_PATH, filename)
            
            #Skip un-labeld images
            true_angle = angle_lookup.get(filename)
            if not true_angle:
                continue

            if DISPLAY_IMAGES:
                print(f"Processing: {image_path}")
            
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load: {image_path}")
                continue
            
            height, width = image.shape[:2]
            
            bbox = (0, 0, width, height) # Fake bounding box for corner detection (the whole image)
            formated_image = {
                "bots": [
                    {
                        'img': image,
                        'bbox': bbox
                    }
                ]
            }
            
            quantized_bots = quantize(formated_image, selected_colors, show=False, is_flipped=1, settings=quant_settings) #Quantize with custom settings
            quantized_img = quantized_bots['bots'][0]['img']

            corner_detection.set_bots(quantized_bots)
            detected_bots_with_data = corner_detection.corner_detection_main(area_threshold)
            
            bbox = detected_bots_with_data[0]['huey']['bbox']
            x, y, w, h = bbox
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            
            total_frames += 1

            # Differs from main
            # Adds to score variable
            if detected_bots_with_data[0]['huey']['orientation'] != None:
                frames_with_orientation += 1
                current_angle_difference = angle_difference(true_angle, detected_bots_with_data[0]['huey']['orientation']) #Use squared difference ?
                total_theta += current_angle_difference
                total_score += math.pow(current_angle_difference, 2) # Score increases by angle difference
            else:
                total_score += math.pow(NO_ORIENTATION_SCORE, 2) # Score increases by arbitrary value

            if DISPLAY_IMAGES:
                print(f"Correct Angle: {angle_lookup.get(filename)}")
                print(f"Calculated Angle: {detected_bots_with_data[0]['huey']['orientation']}")
                draw_manual_arrow(quantized_img, cx, cy, true_angle)
                if detected_bots_with_data[0]['huey']['orientation'] != None:
                    draw_orientation_arrow(quantized_img, detected_bots_with_data)

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
    if DISPLAY_IMAGES:
        cv2.destroyAllWindows()

    return (total_score/total_frames) # Return average score

if __name__ == "__main__":
    print("PRESS 0 TO SWITCH IMAGES AND N TO ITERATE QUANTIZATION SETTINGS")

    orientation_scores = {}

    # for i in range(15, 25, 3):
    #     orientation_scores[i] = test_detect_corners(i, 0.1, 1.0, 1.0, area_threshold=15)

    for i in range(20, 4, -2):
        orientation_scores[i] = test_detect_corners(25, 0.1, 1.0, 1.0, area_threshold=i)

    print(str(orientation_scores))