import os
import time
import threading
from collections import deque

import pandas as pd
import cv2
import numpy as np
from time import perf_counter as ptime

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
WARP_AND_COLOR_PICKING = False   # Re-do warp & color selection
DISPLAY_SCALE = 1.0             # Display frame smaller for selection with 1080p video, 1.0 default
IS_TRANSMITTING = False         # True to send transmissions to live Huey via Arduino
WEAPON_ON = False               # True if weapon motor should be on
SHOW_FRAME = True               # Show camera feed frames
DISPLAY_ANGLES = True           # Only use when SHOW_FRAME is True
IS_ORIGINAL_FPS = True          # Process every captured frame, False -> cap at FRAME_RATE
FRAME_RATE = 120                # FPS used for algo stuff, update to expected FPS on your system.
SHOW_HUD = True                 # Show heads-up display with FPS, speed, turn, frame number
SHOW_QUANTIZED_HUEY = True     # Display the quantized bounding box of Huey in separate window
COLOR_QUANTIZATION = True       # True to use color quantization, should always be True
CAN_RECOVER = True              # True to use recovery
CAMERA_STREAM = False           # True if using live camera stream, False if using a video file
SHEET_RUNTIME = True            # Save runtimes to a spreadsheet and generate a graph (install "Excel Viewer" VS Code extension)
SAVE_BBOXES = False             # Save bounding box images every BBOX_SAVE_FREQUENCY iterations
BBOX_SAVE_FREQUENCY = 10        # How often to save bounding box images (every n iterations)

# MODEL_NAME = "SmallComp"        # Used for Feb comp, best accuracy if you have the compute for it.
# MODEL_NAME = "NanoSizeVariant"    # MAIN MODEL: Use with lower image size for faster performance, not much worse accuracy.
MODEL_NAME = "Nano320Temp"        # Model trained with Huey images from matches, trained at 320 image size

# Image size for object detection model, lower number -> faster, slightly worse accuracy.
# 640 default, 416 fast, must be multiple of 32. Don't go below 320.
OD_IMG_SIZE = 320

# If model can't be found or gives a bug, use convert_models.py to regenerate the model w/ above parameters

folder = os.getcwd() + "/main_files"

# camera_number = folder + "/test_videos/huey_vs_prince.mp4"
# camera_number = folder + "/test_videos/huey_hell.mp4"
camera_number = folder + "/test_videos/orbital_huey.mp4"
# camera_number = 1
# camera_number = 0

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

rs = RuntimeSheet(use=SHEET_RUNTIME)
# ------------------------------ BEFORE THE MATCH ------------------------------

# Threading globals
frame_buffer = deque(maxlen=1)
stop_event = threading.Event()
# Shared state for controls passed from UI thread to Perception thread
shared_state = {"key": None, "flipped": None,
                "paused": False, "skip_frame": False}


def main():
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
            if WEAPON_ON:
                weapon_motor_group.move(1)

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

        # ----------------------------------------------------------------------
        # Define the Perception Pipeline (Runs in Background Thread)
        def perception_pipeline():
            prev = ptime()
            last_frame = 0
            iteration = 0
            start_time = ptime()

            while not stop_event.is_set():
                # Check if source is still open
                if CAMERA_STREAM and (not stream.isOpened() or stream.stopped):
                    break
                if not CAMERA_STREAM and not cap.isOpened():
                    break

                # Handle Pause (Simple spin wait)
                if shared_state["paused"]:
                    if shared_state["skip_frame"]:
                        shared_state["skip_frame"] = False
                        # Proceed to process one frame
                    else:
                        time.sleep(0.05)
                        continue

                time_elapsed = ptime() - prev

                rs.start_iter()
                if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / FRAME_RATE) and (not CAMERA_STREAM or stream.frameCount() > last_frame):
                    prev = ptime()
                    iteration += 1

                    # Save bboxes every BBOX_SAVE_FREQUENCY iterations if SAVE_BBOXES is True
                    if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                        os.makedirs(
                            f"{bb_output_dir}/frame_{iteration}", exist_ok=True)
                        frame_save_dir = os.path.join(
                            bb_output_dir, f"frame_{iteration}")

                    # Logs average of last 10 FPS
                    if SHEET_RUNTIME:
                        if iteration > 11:
                            fps10 = 1.0 / \
                                ((prev - rs.get_row(-10)["Start Time"]) / 10.0)
                        else:
                            fps10 = 1.0 / ((prev - start_time) / iteration)
                        rs.log("FPS10", fps10)
                    else:
                        fps10 = None

                    # Grabs frame from camera thread if using camera stream, otherwise reads from video
                    with rs.log_timing("Frame Read"):
                        if CAMERA_STREAM:
                            ret, frame = stream.read()
                            last_frame = stream.frameCount()
                        else:
                            ret, frame = cap.read()

                        if not ret:
                            print("Failed to capture image" + "\n")
                            break

                    # Get inputs from Shared State
                    key = shared_state["key"]
                    is_flipped = -1 if shared_state["flipped"] else 1

                    # Warp image to homography matrix using maps
                    with rs.log_timing("Warp"):
                        warped_frame = warp_map(frame, map_x, map_y)

                    # 11. Run the Warped Image through Object Detection
                    # Internal timings (Preprocess, Inference, etc.) are handled inside predict()
                    with rs.log_timing("Object Detection"):
                        detected_bots = predictor.predict(warped_frame)

                    if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                        for bot in range(len(detected_bots["bots"])):
                            if detected_bots["bots"][bot]["img"] is not None:
                                cv2.imwrite(
                                    f"{frame_save_dir}/detected_bot_{bot}.png", detected_bots["bots"][bot]["img"])

                    # 11.5 Quantize Colors
                    with rs.log_timing("Color Quantization"):
                        if COLOR_QUANTIZATION:
                            detected_bots = quantize(
                                detected_bots, selected_colors, show=False, is_flipped=is_flipped)

                    if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                        for bot in range(len(detected_bots["bots"])):
                            if detected_bots["bots"][bot]["img"] is not None:
                                cv2.imwrite(
                                    f"{frame_save_dir}/quantized_bot_{bot}.png", detected_bots["bots"][bot]["img"])

                    # 12. Run Object Detection's results through Corner Detection
                    with rs.log_timing("Corner Detection"):
                        corner_detection.set_bots(detected_bots)
                        detected_bots_with_data = corner_detection.corner_detection_main()

                    # Prepare Quantized Huey Image (for display buffer)
                    huey_display_img = None
                    with rs.log_timing("Display Quantized Huey"):
                        if SHOW_QUANTIZED_HUEY and len(detected_bots["bots"]) > 0:
                            try:
                                if detected_bots_with_data and detected_bots_with_data.get("huey") and detected_bots_with_data["huey"].get("bbox") is not None:
                                    huey_bbox = detected_bots_with_data["huey"]["bbox"]
                                    # Find which bot index has this bbox
                                    for i, bot in enumerate(detected_bots["bots"]):
                                        if bot.get("bbox") is not None and np.array_equal(bot["bbox"], huey_bbox):
                                            if bot.get("img") is not None:
                                                huey_display_img = bot["img"]
                                            break
                            except Exception as e:
                                pass

                    with rs.log_timing("Algorithm"):
                        move_dictionary = algorithm.ram_ram(
                            detected_bots_with_data, CAN_RECOVER, fps=FRAME_RATE, key=key)

                    # 14. Transmitting the motor values to Huey's if we're using a live video
                    with rs.log_timing("Transmission"):
                        if IS_TRANSMITTING:
                            speed = move_dictionary["speed"]
                            turn = move_dictionary["turn"]
                            motor_group.move(speed*is_flipped, turn * -1)
                            # Added this post-comp, see if it works?
                            if WEAPON_ON:
                                weapon_motor_group.move(1)

                    # Prepare Main Display Image
                    main_display_img = None
                    with rs.log_timing("Display"):
                        if DISPLAY_ANGLES:
                            warped_frame = predictor.show_predictions(
                                warped_frame, detected_bots)
                            if SHOW_HUD:
                                warped_frame = draw_hud(
                                    warped_frame, fps10=fps10, move_dictionary=move_dictionary, iteration=iteration)

                            # Call display_angles with show=False to get the image without displaying
                            main_display_img = display_angles(detected_bots_with_data, move_dictionary, warped_frame, is_recovering=algorithm.is_recovering, is_backing=algorithm.is_backing,
                                                              against_wall=algorithm.against_wall, moving_forward=algorithm.moving_forward, is_flipped=is_flipped, centroids=corner_detection.centroids, show=False)

                            if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                                cv2.imwrite(
                                    f"{frame_save_dir}/final_image_{iteration}.png", main_display_img)

                        elif SHOW_FRAME:
                            display_frame = warped_frame
                            if SHOW_HUD:
                                display_frame = draw_hud(
                                    display_frame, fps10=fps10, move_dictionary=move_dictionary, iteration=iteration)
                            main_display_img = display_frame

                    # Update Frame Buffer
                    frame_buffer.append({
                        "main": main_display_img,
                        "huey": huey_display_img
                    })

                    rs.dump()

        # Start the Perception Thread
        perception_thread = threading.Thread(
            target=perception_pipeline, daemon=True)
        perception_thread.start()

        # ----------------------------------------------------------------------
        # Display UI Loop (Runs in Main Thread)
        while not stop_event.is_set():
            if frame_buffer:
                frames = frame_buffer[0]

                if frames["main"] is not None and SHOW_FRAME:
                    name = "Battle with Predictions" if DISPLAY_ANGLES else "Bounding boxes (no angles)"
                    cv2.imshow(name, frames["main"])

                if frames["huey"] is not None and SHOW_QUANTIZED_HUEY:
                    cv2.imshow("Quantized Huey", frames["huey"])

            # pollKey handles the GUI event loop
            key = cv2.pollKey()

            if key != -1:
                key_8bit = key & 0xFF
                if key_8bit == ord("q"):
                    stop_event.set()
                elif key_8bit == ord("f"):
                    print("Backup flipped key pressed")
                    if shared_state["flipped"] is None:
                        shared_state["flipped"] = True
                    else:
                        shared_state["flipped"] = not shared_state["flipped"]
                    if shared_state["paused"]:
                        shared_state["skip_frame"] = True
                elif key_8bit == ord("p"):
                    shared_state["paused"] = not shared_state["paused"]
                    shared_state["skip_frame"] = False
                    print(
                        f"Playback {'paused' if shared_state['paused'] else 'resumed'}")
                elif shared_state["paused"]:
                    # Any other key while paused skips one frame
                    shared_state["skip_frame"] = True

            # Pass key to perception thread (resetting it to None if no key pressed is handled by waitKey returning 255)
            shared_state["key"] = key if key != -1 else None

            # Check if thread died
            if not perception_thread.is_alive():
                break

        # Wait for the background perception thread to finish its current iteration and exit
        perception_thread.join()

        if CAMERA_STREAM:
            stream.stop()
        print("============================")
        print("Video finished successfully!")

        if SHOW_FRAME:
            cv2.destroyAllWindows()
            if SHOW_QUANTIZED_HUEY:
                try:
                    cv2.destroyWindow("Quantized Huey")
                except:
                    pass

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")
    except Exception as exception:
        print("UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP:", exception)
    finally:

        # Newbie squadron trial
        try:
            color_df = pd.DataFrame(corner_detection.color_percentage_rows)
            color_df.to_csv("color_output.csv", index=True)
            # color_percentages_graphing.makeGraph()
        except Exception as color_exception:
            print("Data collection failed:", color_exception)

        if IS_TRANSMITTING:  # Motors need to be cleaned up correctly
            try:
                if 'motor_group' in locals():
                    motor_group.stop()
                if 'weapon_motor_group' in locals():
                    weapon_motor_group.stop()
                if 'ser' in locals():
                    ser.cleanup()
            except Exception as motor_exception:
                print("Motor cleanup failed:", motor_exception)

        if CAMERA_STREAM:
            if stream:
                stream.stop()
                cv2.destroyAllWindows()
        elif cap != None:
            cap.release()
            cv2.destroyAllWindows()

        rs.save("itertimes")


if __name__ == "__main__":
    main()
