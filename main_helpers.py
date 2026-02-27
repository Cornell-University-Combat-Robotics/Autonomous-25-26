import cv2
import numpy as np
import math
import time
import torch
import openvino as ov

from algorithm.ram import Ram
from corner_detection.color_picker import ColorPicker
from machine.predict import YoloModel
from transmission.motors import Motor
from transmission.serial_conn import OurSerial
from warp_main import get_homography_mat, warp
from color_quant.quantization import quantize_robot_colors

"""
Gets first frame of the video and returns it. If frame can't be read or video isn't being 
processed will print the problem, and return captured_image as none. 
"""
def key_frame(stream, CAMERA_STREAM):
    captured_image = None
    if stream == None:
            print("Error opening camera stream" + "\n")

    while (CAMERA_STREAM and stream.isOpened() and not stream.stopped) or stream.isOpened():
        ret, frame = stream.read()

        if ret and frame is not None:
            cv2.imshow("Press 'q' to quit. Press '0' to capture the image", frame)
            key = cv2.waitKey(1) & 0xFF  # Check for key press

            if key == ord("q"):  # Press 'q' to quit without capturing
                return captured_image
            elif key == ord("0"):  # Press '0' to capture the image and exit
                captured_image = frame.copy()
                return captured_image
            time.sleep(0.01)
        else:
            print("Failed to read frame" + "\n")
            return captured_image
    cv2.destroyAllWindows()
    return captured_image

def read_prev_homography(captured_image, file_path):
    homography_matrix = []
    homography_matrix_file = file_path
    try:
        with open(homography_matrix_file, "r") as file:
            for line in file:
                row = list(map(float, line.strip().split(", ")))
                homography_matrix.append(row)
        if len(homography_matrix) != 3 or len(homography_matrix[0]) != 3:
            raise ValueError("The file must represent a 3 x 3 matrix.")
        homography_matrix = np.array(
            homography_matrix, dtype=np.float32)
    except Exception as e:
        print(f"Error reading homography_matrix.txt: {e}" + "\n")
        exit(1)
        
    warped_frame = warp(captured_image, homography_matrix)
    return warped_frame, homography_matrix

def make_new_homography(captured_image):
    if captured_image is None:
        print("No image captured. Press '0' to capture image.")
        return
    
    homography_matrix = get_homography_mat(captured_image)
    warped_frame = warp(captured_image, homography_matrix)

    return warped_frame, homography_matrix

def read_prev_colors(file_path):
    selected_colors = []
    selected_colors_file = file_path
    try:
        with open(selected_colors_file, "r") as file:
            for line in file:
                hsv = list(map(int, line.strip().split(", ")))
                selected_colors.append(hsv)
        if len(selected_colors) != 3:
            raise ValueError("The file must contain exactly 3 HSV values.")
    except Exception as e:
        print(f"Error reading selected_colors.txt: {e}" + "\n")
        exit(1)
    return selected_colors

def make_new_colors(output_file_path, warped_frame):
    selected_colors = ColorPicker.pick_colors(warped_frame)
    with open(output_file_path, "w") as file:
        for color in selected_colors:
            file.write(f"{color[0]}, {color[1]}, {color[2]}\n")
    return selected_colors

import platform

def is_coreml_available():
    # 1. Platform Check: CoreML inference only runs on macOS
    # if platform.system() != "Darwin":
    #     return False
    
    # 2. Library & Hardware Check
    try:
        import coremltools as ct
        # get_all_compute_devices() returns a list of hardware the framework can see
        # This will fail or return an empty/CPU-only list if the OS/Framework is broken
        devices = ct.models.MLComputeDevice.get_all_compute_devices()
        return len(devices) > 0
    except (ImportError, AttributeError, Exception):
        # Fails if coremltools isn't installed or if 
        # run on an OS version where the API doesn't exist
        return False

def get_predictor(MODEL_NAME, OD_IMG_SIZE):
    if torch.cuda.is_available():
        print(f"Using {MODEL_NAME} on CUDA for object detection.")
        predictor = YoloModel(MODEL_NAME, "TensorRT", OD_IMG_SIZE, device="cuda")

    elif is_coreml_available() or torch.backends.mps.is_available():
        print(f"Using {MODEL_NAME} with CoreML for object detection.")
        predictor = YoloModel(MODEL_NAME, "CoreML", OD_IMG_SIZE)

    elif torch.backends.mps.is_available():
        print(f"Using {MODEL_NAME} on MPS for object detection.")
        predictor = YoloModel(MODEL_NAME, "PT", OD_IMG_SIZE, device="mps")

    elif ov.Core().get_available_devices() and "CPU" in ov.Core().get_available_devices():
        print(f"Using {MODEL_NAME} with OpenVINO on CPU for object detection.")
        predictor = YoloModel(MODEL_NAME, "OpenVINO", OD_IMG_SIZE)

    else:
        print(f"Using {MODEL_NAME} with ONNX on CPU for object detection.")
        predictor = YoloModel(MODEL_NAME, "ONNX", OD_IMG_SIZE, device="cpu")
    return predictor

def get_motor_groups(JANK_CONTROLLER, speed_motor_channel, turn_motor_channel, weapon_motor_channel):
    # 5.1: Defining Transmission Object if we're using a live video
    ser = OurSerial()
    motor_group = Motor(ser=ser, channel=speed_motor_channel, channel2=turn_motor_channel)
    if JANK_CONTROLLER:
        weapon_motor_group = Motor(ser=ser, channel=weapon_motor_channel, speed=-1)
    else:
        weapon_motor_group = Motor(ser=ser, channel=weapon_motor_channel)
    return ser, motor_group, weapon_motor_group

def first_run(predictor, warped_frame, SHOW_FRAME, corner_detection, selected_colors):
    # 6. Do an initial run of ML and Corner. Initialize Algo
    first_run_ml = predictor.predict(warped_frame, show=SHOW_FRAME)
    first_run_ml = quantize(first_run_ml, selected_colors, show=False, is_flipped=False)
    corner_detection.set_bots(first_run_ml)
    first_run_orientation = corner_detection.corner_detection_main(threshold_set=True)

    if first_run_orientation and first_run_orientation["huey"] and first_run_orientation["enemy"]:
        # Ensure single enemy
        # first_run_orientation["enemy"] = first_run_orientation["enemy"][0] # we just take the first enemy in the list
        algorithm = Ram(bots=first_run_orientation)
        first_move_dictionary = algorithm.ram_ram(first_run_orientation)

        num_housebots = len(first_run_ml["housebot"])
        num_bots = len(first_run_ml["bots"])
        print("Initial Object Detection: " + str(num_housebots) + " housebots, " + str(num_bots) + " bots detected")
        print("Initial Corner Detection Output: " + str(first_run_orientation))
        print("Initial Algorithm Output: " + str(first_move_dictionary))
        
        display_angles(first_run_orientation, first_move_dictionary, warped_frame, True, centroids=corner_detection.centroids)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        algorithm = Ram()
        cv2.imshow("", warped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Warning: Initial detection of Huey and enemy robot failed." + "\n")
    
    return algorithm

def display_angles(detected_bots_with_data, move_dictionary, image, initial_run=False, is_recovering=False, is_backing=False, against_wall="", moving_forward=-1, is_flipped = False,  centroids=[]):
    if is_recovering:
        cv2.putText(image, "RECOVERING", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 255), 2)
    if is_flipped == -1:
        cv2.putText(image, "FLIPPED", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 255, 0), 2)
    if is_backing:
        if moving_forward > 0:
            cv2.putText(image, "FORWARD: " + against_wall, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (67, 150, 255), 2)
        else:
            cv2.putText(image, "BACKWARD: " + against_wall, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (150, 67, 255), 2)
            
    # BLUE line: Huey's Current Orientation according to Corner Detection

    if detected_bots_with_data and detected_bots_with_data["huey"]:
        start_x = int(detected_bots_with_data["huey"]["center"][0])
        start_y = int(detected_bots_with_data["huey"]["center"][1])

        cv2.putText(image, "HUEY", (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Huey's corner points
        x_shift = int(detected_bots_with_data["huey"]['bbox'][0][0])
        y_shift = int(detected_bots_with_data["huey"]['bbox'][0][1])

        for i in range(len(centroids)):
            color = (255, 255, 0) if i == 0 else (0, 255, 255)
            for p in centroids[i]:
                cv2.circle(image, (p[0] + x_shift, p[1] + y_shift), 8, color, -1)
    
        if detected_bots_with_data["huey"]["orientation"] is not None:
            orientation_degrees = detected_bots_with_data["huey"]["orientation"]

            # Components of current front arrow
            dx = np.cos(math.pi / 180 * orientation_degrees)
            dy = -1 * np.sin(math.pi / 180 * orientation_degrees)

            # Huey's center

            end_point = (int(start_x + 300 * dx), int(start_y + 300 * dy))
            cv2.arrowedLine(image, (start_x, start_y), end_point, (255, 0, 0), 2)

            # RED line: Huey's Desired Orientation according to Algorithm
            if move_dictionary and (move_dictionary["turn"]):
                turn = move_dictionary["turn"] # angle in degrees / 180
                new_orientation_degrees = orientation_degrees + (turn * 180)

                # Components of predicted turn
                dx = np.cos(math.pi * new_orientation_degrees / 180)
                dy = -1 * np.sin(math.pi * new_orientation_degrees / 180)

                end_point = (int(start_x + 300 * dx), int(start_y + 300 * dy))
                cv2.arrowedLine(image, (start_x, start_y), end_point, (0, 0, 255), 2)

    if initial_run:
        cv2.imshow("Initial Run: Battle with Predictions. Press '0' to continue", image)
    else:
        cv2.imshow("Battle with Predictions", image)
    
    return image

    # cv2.waitKey(1)

def initialize_quantization():
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    _ = cv2.cvtColor(dummy, cv2.COLOR_BGR2LAB)
    _ = cv2.cvtColor(dummy, cv2.COLOR_BGR2HSV)

def quantize(detected_bots, selected_colors, show, is_flipped=False):
    if(is_flipped == 1):
        threshold = 18
    else:
        threshold = 22
    # threshold = 30

    colors_hsv_1x = np.array(selected_colors).reshape(1, -1, 3)

    # OpenCV expects uint8 or float32, not int32
    if colors_hsv_1x.dtype != np.uint8:
        colors_hsv_1x = np.clip(colors_hsv_1x, 0, 255).astype(np.uint8)

    bgr_colors_1x = cv2.cvtColor(colors_hsv_1x, cv2.COLOR_HSV2BGR)
    
    bgr_colors = bgr_colors_1x.reshape(-1, 3)  # (N_colors, 3)
    for bot in detected_bots["bots"]:
        bot["img"] = quantize_robot_colors(bot["img"], bgr_colors, thresh_lab=threshold,keep_background=False, show=show)

    return detected_bots

def draw_hud(image, fps10=None, move_dictionary=None, iteration=None, playback_speed=None):
    """
    Draw a heads-up display (HUD) showing real-time metrics on the image.
    
    Args:
        image: The image to draw the HUD on
        fps10: Rolling average FPS (10-frame average)
        move_dictionary: Dictionary containing 'speed' and 'turn' values
        iteration: Current frame iteration number
        playback_speed: Current playback speed multiplier (0.05 to 1.0)
    
    Returns:
        The image with HUD information drawn on it
    """
    if image is None:
        return image
    
    # HUD display configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (0, 255, 0)  # Green color in BGR
    bg_color = (0, 0, 0)  # Black background
    x_offset = 10
    y_offset = 30
    line_height = 28
    
    # Calculate number of lines to display
    num_lines = 4
    if playback_speed is not None:
        num_lines += 1
    
    # Create semi-transparent background for HUD
    hud_height = 10 + (num_lines * line_height)
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (250, hud_height), bg_color, -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    line_num = 0
    
    # Display FPS
    if fps10 is not None:
        fps_text = f"FPS: {fps10:.1f}"
        cv2.putText(image, fps_text, (x_offset, y_offset + line_num * line_height), font, font_scale, text_color, thickness)
        line_num += 1
    
    # Display Speed
    if move_dictionary is not None and "speed" in move_dictionary:
        speed_text = f"Speed: {move_dictionary['speed']:.2f}"
        cv2.putText(image, speed_text, (x_offset, y_offset + line_num * line_height), font, font_scale, text_color, thickness)
        line_num += 1
    
    # Display Turn
    if move_dictionary is not None and "turn" in move_dictionary:
        turn_text = f"Turn: {move_dictionary['turn']:.2f}"
        cv2.putText(image, turn_text, (x_offset, y_offset + line_num * line_height), font, font_scale, text_color, thickness)
        line_num += 1
    
    # Display Iteration (frame number)
    if iteration is not None:
        iter_text = f"Frame: {iteration}"
        cv2.putText(image, iter_text, (x_offset, y_offset + line_num * line_height), font, font_scale, text_color, thickness)
        line_num += 1
    
    # Display Playback Speed
    if playback_speed is not None:
        speed_mult_text = f"Speed: {playback_speed:.2f}x"
        cv2.putText(image, speed_mult_text, (x_offset, y_offset + line_num * line_height), font, font_scale, text_color, thickness)
    
    return image