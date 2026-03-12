import os
import time
import threading
from collections import deque

import pandas as pd
import cv2
import numpy as np
from time import perf_counter as ptime
import dearpygui.dearpygui as dpg

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

# ------------------------------ GLOBAL VARIABLES ------------------------------

# MATT_LAPTOP = False           # Deprecated, matt laptop handled by torch device checks
JANK_CONTROLLER = False         # Deprecated, True if using backup controller?
WARP_AND_COLOR_PICKING = True   # Re-do warp & color selection
# Display frame smaller for selection with 1080p video, 1.0 default
DISPLAY_SCALE = 0.5
IS_TRANSMITTING = False         # True to send transmissions to live Huey via Arduino
WEAPON_ON = False               # True if weapon motor should be on
SHOW_FRAME = True               # Show camera feed frames
DISPLAY_ANGLES = True           # Only use when SHOW_FRAME is True
# Process every captured frame, False -> cap at FRAME_RATE
IS_ORIGINAL_FPS = False
# FPS used for algo stuff, update to expected FPS on your system.
FRAME_RATE = 120
# Show heads-up display with FPS, speed, turn, frame number
SHOW_HUD = True
# Display the quantized bounding box of Huey in separate window
SHOW_QUANTIZED_HUEY = True
# True to use color quantization, should always be True
COLOR_QUANTIZATION = True
CAN_RECOVER = False              # True to use recovery
# True if using live camera stream, False if using a video file
CAMERA_STREAM = False
# Save runtimes to a spreadsheet and generate a graph (install "Excel Viewer" VS Code extension)
SHEET_RUNTIME = True
# Save bounding box images every BBOX_SAVE_FREQUENCY iterations
SAVE_BBOXES = False
# How often to save bounding box images (every n iterations)
BBOX_SAVE_FREQUENCY = 10

# MODEL_NAME = "SmallComp"        # Used for Feb comp, best accuracy if you have the compute for it.
# MODEL_NAME = "NanoSizeVariant"    # MAIN MODEL: Use with lower image size for faster performance, not much worse accuracy.
# Model trained with Huey images from matches, trained at 320 image size
MODEL_NAME = "Nano320Temp"

# Image size for object detection model, lower number -> faster, slightly worse accuracy.
# 640 default, 416 fast, must be multiple of 32. Don't go below 320.
OD_IMG_SIZE = 320

# If model can't be found or gives a bug, use convert_models.py to regenerate the model w/ above parameters

folder = os.getcwd() + "/main_files"

# camera_number = folder + "/test_videos/huey_vs_prince.mp4"
# camera_number = folder + "/test_videos/huey_hell.mp4"
# camera_number = folder + "/test_videos/orbital_huey.mp4"
camera_number = folder + "/test_videos/cicero_corners_bzone.mov"
# camera_number = 1
# camera_number = 0

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

rs = RuntimeSheet(use=SHEET_RUNTIME)

def main():
    # 1. Setup DPG
    dpg.create_context()
    dpg.create_viewport(title='Huey Dashboard', width=800, height=800)

    # 2. Setup Texture Registry
    # Pre-allocate a 'blank' array of the correct size (e.g., 640x480)
    width, height = 700, 700
    # DPG textures are flat arrays of floats (R, G, B, A)
    init_data = np.zeros((height, width, 4), dtype=np.float32)

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width=width, height=height, 
                            default_value=init_data.flatten(), 
                            tag="camera_texture", 
                            format=dpg.mvFormat_Float_rgba)

    # 3. Setup UI
    with dpg.window(label="Camera Feed"):
        dpg.add_image("camera_texture")

    dpg.setup_dearpygui()
    dpg.show_viewport()


    #-------------REGULAR INIT CODE---------------

    stream = None
    try:
        # 1. Start the capturing frame from the camera or pre-recorded video
        # 2. Capture initial frame by pressing '0'
        if CAMERA_STREAM:
            stream = CameraStream(camera_number).start()
            captured_image = key_frame(
                stream, CAMERA_STREAM, selection_scale=DISPLAY_SCALE)
        else:
            cap = cv2.VideoCapture(camera_number)
            captured_image = key_frame(
                cap, CAMERA_STREAM, selection_scale=DISPLAY_SCALE)

        # 3. Use the initial frame to get a new Homography Matrix and new colors
        if WARP_AND_COLOR_PICKING:
            warped_frame, homography_matrix = make_new_homography(
                captured_image, selection_scale=DISPLAY_SCALE)
            selected_colors = make_new_colors(
                folder + "/selected_colors.txt", warped_frame)
        # 3. Or use the previously saved Homography Matrix and colors from the txt file
        else:
            warped_frame, homography_matrix = read_prev_homography(
                captured_image, folder + "/homography_matrix.txt")
            selected_colors = read_prev_colors(folder + "/selected_colors.txt")

        # Build warp maps from homography matrix for faster warping in the main loop
        map_x, map_y = get_warp_maps(homography_matrix)

        # Initialize color quantization cv2
        if COLOR_QUANTIZATION:
            initialize_quantization()

        # Get predictor, if anything goes wrong here, call Aaron #TODO: Document better
        predictor = get_predictor(MODEL_NAME, OD_IMG_SIZE)

        # Initialize corner detection
        corner_detection = RobotCornerDetection(selected_colors, False, False)

        # Initialize transmission TODO: Figure out whether we need weapon_motor_group and JANK_CONTROLLER
        if IS_TRANSMITTING:
            ser, motor_group, weapon_motor_group = get_motor_groups(
                JANK_CONTROLLER, speed_motor_channel, turn_motor_channel, weapon_motor_channel)
            # if WEAPON_ON:
            #     weapon_motor_group.move(1)

        cv2.destroyAllWindows()

        # Initialize algorithm
        if WARP_AND_COLOR_PICKING:
            algorithm = first_run(predictor, warped_frame,
                                  SHOW_FRAME, corner_detection, selected_colors)
        else:
            algorithm = Ram()

        # Initialize BBox save directory
        if SAVE_BBOXES:
            if not os.path.exists("bbox_output"):
                os.makedirs("bbox_output")
            # Make a new directory in bbox_output based on time.time
            bb_output_dir = f"bbox_output/{int(time.time())}"
            os.makedirs(bb_output_dir)
        else:
            bb_output_dir = None

        # ----------------------------------------------------------------------
        # 8. Match begins
        if CAMERA_STREAM:
            if stream.isOpened() == False:
                print("Error opening video file" + "\n")
        else:
            if cap.isOpened() == False:
                print("Error opening video file" + "\n")

    
    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")


    # 4. The Render Loop
    cap = cv2.VideoCapture(camera_number) # Your Elgato or local cam

    # Main loop
    while dpg.is_dearpygui_running():
        ret, frame = cap.read()
        if ret:
            # CONVERT: BGR -> RGBA and normalize to 0.0-1.0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.resize(frame, (width, height)) # Ensure it matches texture size
            data = frame.astype(np.float32) / 255.0
            
            # BLAST: Update the texture data on the GPU
            dpg.set_value("camera_texture", data.flatten())

        dpg.render_dearpygui_frame()

    cap.release()
    dpg.destroy_context()



if __name__ == "__main__":
    main()
