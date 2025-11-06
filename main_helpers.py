import cv2
import numpy as np
import math

from Algorithm.ram import Ram
from corner_detection.color_picker import ColorPicker
from machine.predict import YoloModel
from transmission.motors import Motor
from transmission.serial_conn import OurSerial
from warp_main import get_homography_mat, warp

"""
Gets first frame of the video and returns it. If frame can't be read or video isn't being 
processed will print the problem, and return captured_image as none. 
"""
def key_frame(cap):
    captured_image = None
    if cap.isOpened() == False:
            print("Error opening video file" + "\n")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret and frame is not None:
            cv2.imshow("Press 'q' to quit. Press '0' to capture the image", frame)
            key = cv2.waitKey(1) & 0xFF  # Check for key press

            if key == ord("q"):  # Press 'q' to quit without capturing
                return captured_image
            elif key == ord("0"):  # Press '0' to capture the image and exit
                captured_image = frame.copy()
                return captured_image
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
        
    warped_frame = warp(captured_image, homography_matrix, 700, 700)
    return warped_frame, homography_matrix

def make_new_homography(captured_image):
    if captured_image is None:
        print("No image captured. Press '0' to capture image.")
        return
    
    homography_matrix = get_homography_mat(captured_image, 700, 700)
    warped_frame = warp(captured_image, homography_matrix, 700, 700)

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

def get_predictor(MATT_LAPTOP):
    if MATT_LAPTOP:
        predictor = YoloModel("250v12best", "TensorRT", device="cuda")
    else:
        predictor = YoloModel("100epoch11", "PT", device="mps")
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

def first_run(predictor, warped_frame, SHOW_FRAME, corner_detection):
    # 6. Do an initial run of ML and Corner. Initialize Algo
    first_run_ml = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
    corner_detection.set_bots(first_run_ml)
    
    print("WHAT????")
    first_run_orientation = corner_detection.corner_detection_main()
    print("HELLO?????")

    print("first_run_orientation: " + str(first_run_orientation))
    print("1")
    print("first_run_orientation['huey']: " + str(first_run_orientation["huey"]))
    print("2")
    print("first_run_orientation['enemy']:" + str(first_run_orientation["enemy"]))
    print("it failed before this")

    if first_run_orientation and first_run_orientation["huey"] and first_run_orientation["enemy"]:
        # Ensure single enemy
        # first_run_orientation["enemy"] = first_run_orientation["enemy"][0] # we just take the first enemy in the list
        print("ENTERED IF!!!")
        algorithm = Ram(bots=first_run_orientation)
        print("INITIALIZED RAMMMMMMMMMMMM")
        first_move_dictionary = algorithm.ram_ram(first_run_orientation)

        num_housebots = len(first_run_ml["housebot"])
        num_bots = len(first_run_ml["bots"])
        print("Initial Object Detection: " + str(num_housebots) +
                " housebots, " + str(num_bots) + " bots detected")
        print("Initial Corner Detection Output: " +
                str(first_run_orientation))
        print("Initial Algorithm Output: " +
                str(first_move_dictionary))
        
        display_angles(first_run_orientation, first_move_dictionary, warped_frame, True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        algorithm = Ram()
        cv2.imshow("", warped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Warning: Initial detection of Huey and enemy robot failed." + "\n")
    
    return algorithm

def display_angles(detected_bots_with_data, move_dictionary, image, initial_run=False):
    # BLUE line: Huey's Current Orientation according to Corner Detection

    if detected_bots_with_data and detected_bots_with_data["huey"] and detected_bots_with_data["huey"]["orientation"] is not None:
        orientation_degrees = detected_bots_with_data["huey"]["orientation"]

        print("----- orientation_degrees: " + str(orientation_degrees))

        # Components of current front arrow
        dx = np.cos(math.pi / 180 * orientation_degrees)
        dy = -1 * np.sin(math.pi / 180 * orientation_degrees)

        # Huey's center
        start_x = int(detected_bots_with_data["huey"]["center"][0])
        start_y = int(detected_bots_with_data["huey"]["center"][1])

        end_point = (int(start_x + 300 * dx), int(start_y + 300 * dy))
        cv2.arrowedLine(image, (start_x, start_y), end_point, (255, 0, 0), 2)

        # RED line: Huey's Desired Orientation according to Algorithm
        if move_dictionary and (move_dictionary["turn"]):
            turn = move_dictionary["turn"] # angle in degrees / 180
            new_orientation_degrees = orientation_degrees + (turn * 180)
            print("----- new_orientation_degrees: " + str(new_orientation_degrees))

            # Components of predicted turn
            dx = np.cos(math.pi * new_orientation_degrees / 180)
            dy = -1 * np.sin(math.pi * new_orientation_degrees / 180)

            end_point = (int(start_x + 300 * dx), int(start_y + 300 * dy))
            cv2.arrowedLine(image, (start_x, start_y), end_point, (0, 0, 255), 2)

    if initial_run:
        cv2.imshow("Initial Run: Battle with Predictions. Press '0' to continue", image)
    else:
        cv2.imshow("Battle with Predictions", image)
    cv2.waitKey(1)
