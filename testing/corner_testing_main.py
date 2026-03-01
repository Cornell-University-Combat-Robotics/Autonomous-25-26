
import os
import sys
import cv2
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from corner_testing_helpers import draw_orientation_arrow
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
folder = os.getcwd() + "/testing"

# Valid image file extensions
IMAGE_EXTENSIONS = (".png")

first_heuy_path = os.path.join(FOLDER_PATH, "1.png")
first_huey = cv2.imread(first_heuy_path)
selected_colors = make_new_colors(folder + "/selected_colors.txt", first_huey)
# print(selected_colors)

corner_detection = RobotCornerDetection(selected_colors, False, False)

def detect_corners(quant_settings=False):
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
            
            initialize_quantization()
            quantized_bots = quantize(formated_image, selected_colors, show=False, is_flipped=False, settings=quant_settings)
            quantized_img = quantized_bots['bots'][0]['img']

            corner_detection.set_bots(quantized_bots)
            detected_bots_with_data = corner_detection.corner_detection_main()

            print(detected_bots_with_data)

            if SAMPLE:
                if detected_bots_with_data['huey']['orientation'] != None:
                    draw_orientation_arrow(quantized_img, detected_bots_with_data)
                cv2.imshow("Image Viewer", quantized_img)
                print(f"Showing: {filename} (quantized)")

                key = cv2.waitKey(0)  # Wait for key press
                if key == ord('n'):   # Press 'q' to quit early
                    return

    cv2.destroyAllWindows()

print("PRESS 0 TO SWITCH IMAGES AND N TO CHANGE QUANTIZATION SETTINGS")
for i in range(0, 80, 10):
    detect_corners(quant_settings={
        "threshold": i
    })