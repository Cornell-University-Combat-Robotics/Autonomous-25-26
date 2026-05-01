# INCREDIBLY IMPORTANT: if you want to run this file you need to run it from outside 
# of the testing directory or else the predict breaks so run python testing/collect_huey_imgs.py

import os
import sys
import json
import threading
import pandas as pd
import cv2
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datetime import datetime
from camera_stream import CameraStream
from runtimesheet.runtimesheet import RuntimeSheet
import matplotlib.pyplot as plt
from algorithm.ram import Ram
from corner_detection.corner_detection import RobotCornerDetection
from main_helpers import (
    display_angles,
    draw_hud,
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
from warp_main import warp
from warp_main import get_warp_maps
from warp_main import warp_map

folder = os.getcwd() + "/main_files"

CAMERA_STREAM = False               # Keep false when running on video
DISPLAY_SCALE = 1.0                 # Display frame smaller for selection with 1080p video, 1.0 default
OD_IMG_SIZE = 320                   # Image size for object detection model, lower number -> faster, slightly worse accuracy.
MODEL_NAME = "Nano320Temp"          # Model trained with Huey images from matches, trained at 320 image size
COLLECTION_FREQ = 10                # Collect image every x frames
VIDEO_NAME = "huey_hell.mp4"        # Name of video being used in main files

# time_string = datetime.now().strftime("%H:%M:%S")
time_string = str(int(time.time()))

camera_number = folder + "/test_videos/" + VIDEO_NAME
save_folder = os.path.join("testing", "testing_data", VIDEO_NAME + time_string)
os.makedirs(save_folder)

def main():
    predictor = get_predictor(MODEL_NAME, OD_IMG_SIZE)

    cap = cv2.VideoCapture(camera_number)
    captured_image = key_frame(
        cap, CAMERA_STREAM, selection_scale=DISPLAY_SCALE)

    stop_event = threading.Event()

    warped_frame, homography_matrix = make_new_homography(
        captured_image, selection_scale=DISPLAY_SCALE)
    selected_colors = make_new_colors(
        folder + "/selected_colors.txt", warped_frame)

    initialize_quantization()

    frame_number = 0
    while not stop_event.is_set():
        map_x, map_y = get_warp_maps(homography_matrix)
        # initialize_quantization()
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image" + "\n")
            break

        warped_frame = warp_map(frame, map_x, map_y)
        
        detected_bots = predictor.predict(warped_frame)
        detected_bots = quantize(detected_bots, selected_colors, show=False, is_flipped=False)

        corner_detection = RobotCornerDetection(selected_colors, False, False)
        corner_detection.set_bots(detected_bots)

        detected_bots_with_data = corner_detection.corner_detection_main()
        print(detected_bots_with_data)

        # Differs from main
        # Here we collect an unquantized image of Huey using the bounding box we get from corner_detection_main()
        huey_bbox=detected_bots_with_data['huey']['bbox']
        x1, y1 = int(huey_bbox[0][0]), int(huey_bbox[0][1])
        x2, y2 = int(huey_bbox[1][0]), int(huey_bbox[1][1])
        cropped = warped_frame[y1:y2, x1:x2]

        if frame_number % COLLECTION_FREQ == 0:
            file_save_path = os.path.join(save_folder, f"{frame_number}.png")
            cv2.imwrite(file_save_path, cropped)
            print(f"Saved crop: x1={x1}, y1={y1}, x2={x2}, y2={y2}, shape={cropped.shape}")
        frame_number += 1

if __name__ == "__main__":
    main()