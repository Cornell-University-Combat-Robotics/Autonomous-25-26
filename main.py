import math
import os
import time
import traceback
import sys

import cv2
import numpy as np
from line_profiler import profile
import math
import matplotlib.pyplot as plt

from Algorithm.ram import Ram
from corner_detection.color_picker import ColorPicker
from arrow.arrow import Arrow
from machine.predict import RoboflowModel, YoloModel
from transmission.motors import Motor
from transmission.serial_conn import OurSerial
from warp_main import get_homography_mat, warp
from vid_and_img_processing.unfisheye import unfish
from vid_and_img_processing.unfisheye import prepare_undistortion_maps

# ------------------------------ GLOBAL VARIABLES ------------------------------

# Set True if using Matt's Laptop
MATT_LAPTOP = False

JANK_CONTROLLER = False

# Set True to optimize for competition, removing all visuals
COMP_SETTINGS = False

# Set True to print outputs for Corner Detection and Algo
PRINT = True

# Save times to an array for plotting
TIMING = False

# Set True to redo warp and picking Huey's main color, front and back corners
WARP_AND_COLOR_PICKING = True

# Set True when testing with a live Huey and not a pre-filmed video
IS_TRANSMITTING = False

# True to display current and future orientation angles for each iteration
SHOW_FRAME = True
DISPLAY_ANGLES = True

# Set True to process every single frame the camera captures
IS_ORIGINAL_FPS = False

UNFISHEYE = True

if COMP_SETTINGS:
    SHOW_FRAME = False
    DISPLAY_ANGLES = False
    PRINT = False
    MATT_LAPTOP = True

if not SHOW_FRAME:
    DISPLAY_ANGLES = False

folder = os.getcwd() + "/new_main_files"
test_videos_folder = folder
resize_factor = 0.8
frame_rate = 50

map1 = np.load('vid_and_img_processing/700xmap1.npy')
map2 = np.load('vid_and_img_processing/700xmap2.npy')

BACK_UP_TIME = 0.5
start_back_up_time = 0

# camera_number = 1
# camera_number = 701
# camera_number = test_videos_folder + "/crude_rot_huey.mp4"
# camera_number = test_videos_folder + "/huey_duet_demo.mp4"
# camera_number = test_videos_folder + "/huey_demo2.mp4"
# camera_number = test_videos_folder + "/huey_demo3.mp4"
# camera_number = test_videos_folder + "/only_huey_demo.mp4"
# camera_number = test_videos_folder + "/only_enemy_demo.mp4"
# camera_number = test_videos_folder + "/green_huey_demo.mp4"
# camera_number = test_videos_folder + "/when_i_throw_it_back_huey.mp4"
camera_number = test_videos_folder + "/huey_small.mp4"
# camera_number = test_videos_folder + "/yellow_huey_demo.mp4"
# camera_number = test_videos_folder + "/warped_no_huey.mp4"
# camera_number = test_videos_folder + "/flippy_huey.mp4"
# camera_number = test_videos_folder + "/nhrl_arena.mp4"


if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

# ------------------------------ BEFORE THE MATCH ------------------------------


@profile
def main():

    if TIMING:
        t_cap = []
        t_fish = []
        t_warp = []
        t_turn = []
        t_predict = []
        t_whole = []

    try:
        # 1. Start the capturing frame from the camera or pre-recorded video
        # 2. Capture initial frame by pressing '0'
        cap = cv2.VideoCapture(camera_number)
        captured_image = None

        if cap.isOpened() == False:
            print("Error opening video file" + "\n")

        while cap.isOpened():
            ret, frame = cap.read()

            if ret and frame is not None:
                cv2.imshow("Press 'q' to quit. Press '0' to capture the image", frame)
                key = cv2.waitKey(1) & 0xFF  # Check for key press

                if key == ord("q"):  # Press 'q' to quit without capturing
                    break
                elif key == ord("0"):  # Press '0' to capture the image and exit
                    captured_image = frame.copy()
                    resized_image = cv2.resize(
                        captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
                    h, w = resized_image.shape[:2]
                    map1, map2 = prepare_undistortion_maps(w, h)
                    cv2.imwrite(folder + "/captured_image.png", captured_image)
                    break
            else:
                print("Failed to read frame" + "\n")
                break
        cv2.destroyAllWindows()

        # 3. Use the initial frame to get the Homography Matrix
        if WARP_AND_COLOR_PICKING:
            if captured_image is None:
                print(
                    "No image was captured. Please press '0' to capture an image before continuing.")
                return
            resized_image = cv2.resize(
                captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
            cv2.imwrite(folder + "/resized_image.png", resized_image)
            h, w = resized_image.shape[:2]
            map1, map2 = prepare_undistortion_maps(w, h)
            if (UNFISHEYE):
                resized_image = unfish(resized_image, map1, map2)
            homography_matrix = get_homography_mat(resized_image, 700, 700)

            warped_frame = warp(resized_image, homography_matrix, 700, 700)
            cv2.imwrite(folder + "/warped_frame.png", warped_frame)

            # 3.2 Part 2. ColorPicker: Manually picking colors for Huey, front and back corners
            image_path = folder + "/warped_frame.png"
            output_file = folder + "/selected_colors.txt"
            selected_colors = ColorPicker.pick_colors(image_path)
            with open(output_file, "w") as file:
                for color in selected_colors:
                    file.write(f"{color[0]}, {color[1]}, {color[2]}\n")
            print(
                f"Selected colors have been saved to '{output_file}'." + "\n")
        # 3. Or use the previously saved homography matrix from the txt file
        else:
            homography_matrix = []
            homography_matrix_file = folder + "/homography_matrix.txt"
            try:
                with open(homography_matrix_file, "r") as file:
                    for line in file:
                        row = list(map(float, line.strip().split(" ")))
                        homography_matrix.append(row)
                if len(homography_matrix) != 3 or len(homography_matrix[0]) != 3:
                    raise ValueError("The file must represent a 3 x 3 matrix.")
                homography_matrix = np.array(
                    homography_matrix, dtype=np.float32)
            except Exception as e:
                print(f"Error reading homography_matrix.txt: {e}" + "\n")
                exit(1)

        # 4. Reading the HSV values for Huey, front and back corners from a text file
        selected_colors = []
        selected_colors_file = folder + "/selected_colors.txt"
        try:
            with open(selected_colors_file, "r") as file:
                for line in file:
                    hsv = list(map(int, line.strip().split(", ")))
                    selected_colors.append(hsv)
            if len(selected_colors) != 4:
                raise ValueError("The file must contain exactly 4 HSV values.")
        except Exception as e:
            print(f"Error reading selected_colors.txt: {e}" + "\n")
            exit(1)

        # 5. Defining all subsystem objects: ML, Corner, Algorithm
        # Defining Roboflow Machine Learning Model Object
        # predictor = RoboflowModel()
        if MATT_LAPTOP:
            predictor = YoloModel("250v12best", "TensorRT", device="cuda")
        else:
            predictor = YoloModel("250v12best", "PT", device="mps")

        # Defining Corner Detection Object
        arrow = Arrow(selected_colors, False)

        # Defining Ram Ram Algorithm Object
        algorithm = None

        if IS_TRANSMITTING:
            # 5.1: Defining Transmission Object if we're using a live video
            ser = OurSerial()
            # speed_motor_group = Motor(ser=ser, channel=speed_motor_channel)
            # turn_motor_group = Motor(ser=ser, channel=turn_motor_channel)
            motor_group = Motor(
                ser=ser, channel=speed_motor_channel, channel2=turn_motor_channel)
            if JANK_CONTROLLER:
                weapon_motor_group = Motor(
                    ser=ser, channel=weapon_motor_channel, speed=-1)
            else:
                weapon_motor_group = Motor(
                    ser=ser, channel=weapon_motor_channel)

        cv2.destroyAllWindows()

        if WARP_AND_COLOR_PICKING:
            # 6. Do an initial run of ML and Corner. Initialize Algo
            first_run_ml = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
            print("first_run_ml: " + str(first_run_ml))
            arrow.set_bots(first_run_ml)
            first_run_orientation = arrow.arrow_main([b["img"] for b in first_run_ml["bots"] if b.get("img") is not None])
            print("first_run_orientation: " + str(first_run_orientation))

            if first_run_orientation and first_run_orientation["huey"] and first_run_orientation["enemy"]:
                # Ensure single enemy
                print("HELLO 1")
                first_run_orientation["enemy"] = first_run_orientation["enemy"][0] # we just take the first enemy in the list
                algorithm = Ram(bots=first_run_orientation)
                first_move_dictionary = algorithm.ram_ram(first_run_orientation)
                print("HELLO 2")

                num_housebots = len(first_run_ml["housebot"])
                num_bots = len(first_run_ml["bots"])
                print("Initial Object Detection: " + str(num_housebots) +
                        " housebots, " + str(num_bots) + " bots detected")
                print("Initial Corner Detection Output: " +
                        str(first_run_orientation))
                print("Initial Algorithm Output: " +
                        str(first_move_dictionary))
            else:
                algorithm = Ram()
                cv2.imshow("", warped_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(
                    "Warning: Initial detection of Huey and enemy robot failed." + "\n")
        else:
            algorithm = Ram()

        # ----------------------------------------------------------------------

        # 8. Match begins
        if cap.isOpened() == False:
            print("Error opening video file" + "\n")

        prev = 0
        t0 = time.perf_counter()

        while cap.isOpened():
            time_elapsed = time.perf_counter() - prev

            # 10. Warp image using the Homography Matrix
            if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:

                # 9. Frames are being capture by the camera/pre-recorded video
                if TIMING:
                    t1 = time.perf_counter()
                ret, frame = cap.read()
                if TIMING:
                    t_cap.append(time.perf_counter()-t1)
                if not ret:
                    # If frame capture fails, break the loop
                    print("Failed to capture image" + "\n")
                    break

                if SHOW_FRAME:
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q on keyboard to exit
                        print("exit" + "\n")
                        break

            # 10. Warp image using the Homography Matrix
            if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:
                global start_back_up_time
                prev = time.perf_counter()
                frame = cv2.resize(
                    frame, (0, 0), fx=resize_factor, fy=resize_factor)
                if (UNFISHEYE):
                    frame = unfish(frame, map1, map2)
                warped_frame = warp(frame, homography_matrix, 700, 700)

                if TIMING:
                    t2 = time.perf_counter()
                # 11. Run the Warped Image through Object Detection
                detected_bots = predictor.predict(
                    warped_frame, show=SHOW_FRAME, track=True)
                if TIMING:
                    t_predict.append(time.perf_counter()-t2)

                # 12. Run Object Detection's results through Corner Detection
                arrow_dictionary = arrow.arrow_main([b["img"] for b in detected_bots["bots"] if b.get("img") is not None])
                detected_bots_with_data = detected_bots

                if PRINT:
                    print("CORNER DETECTION: " + str(detected_bots_with_data))

                print("detected_robots_with_data['enemy']:" + str(arrow_dictionary["enemy"]))
                if arrow_dictionary and arrow_dictionary["enemy"]:
                    arrow_dictionary["enemy"] = arrow_dictionary["enemy"][0] # assumes first enemy roboy
                print("HELLO RAHHHH")
                
                move_dictionary = algorithm.ram_ram(
                    arrow_dictionary)
                print("EXITED RAM!!!!!!!!!!!!!!!!!!! OOO WOAH~")
                
                if arrow_dictionary and arrow_dictionary["huey"]:
                    if PRINT:
                        print("ALGORITHM: " + str(move_dictionary))
                    # if DISPLAY_ANGLES:
                    #     display_angles(arrow_dictionary,
                    #                     move_dictionary, warped_frame)
                    # 14. Transmitting the motor values to Huey's if we're using a live video
                    if IS_TRANSMITTING:
                        speed = move_dictionary["speed"]
                        turn = move_dictionary["turn"]

                        tt = time.perf_counter()
                        if turn * -1 > 0:
                            motor_group.move(speed * 0.8, turn * -1 * 0.55 + 0.2)
                        else:
                            motor_group.move(speed * 0.8, turn * -1 * 0.55 - 0.2)
                        t_turn.append(time.perf_counter() - tt)
                #     elif DISPLAY_ANGLES:
                #         display_angles(arrow_dictionary,
                #                     None, warped_frame)
                # elif DISPLAY_ANGLES:
                #     display_angles(None, None, warped_frame)

                if SHOW_FRAME and not DISPLAY_ANGLES:
                    print("RAHHHH")
                    cv2.imshow("Bounding boxes (no angles)", warped_frame)

                if TIMING:
                    t_whole.append(time.perf_counter()-t0)
                    t0 = time.perf_counter()

        cap.release()
        print("============================")
        print("Video finished successfully!")

        if SHOW_FRAME:
            cv2.destroyAllWindows()











    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")
    except Exception as exception:
        print("UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP:", exception)
        # traceback.print_exc(file=sys.stdout) #uncomment to see stack trace
    finally:
        if IS_TRANSMITTING:
            # Motors need to be cleaned up correctly
            try:
                # We stop or clean up objects that actually exist in the current scope
                if 'motor_group' in locals():
                    motor_group.stop()
                if 'ser' in locals():
                    ser.cleanup()
            except Exception as motor_exception:
                print("Motor cleanup failed:", motor_exception)

        if 'cap' in locals():
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run using 'python -m kernprof -lvr --unit 1e-3 main.py' for debugging
    main()
