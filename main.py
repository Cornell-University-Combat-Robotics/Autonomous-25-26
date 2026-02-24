import os
import time

import pandas as pd
import cv2
import numpy as np
import torch
from time import perf_counter as ptime

from camera_stream import CameraStream
from runtimesheet import RuntimeSheet
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
    quantize,
    get_next_unique_frame
)
from warp_main import warp
from warp_main import get_warp_maps
from warp_main import warp_map

# ------------------------------ GLOBAL VARIABLES ------------------------------

# MATT_LAPTOP = False           # Deprecated, matt laptop handled by torch device checks
JANK_CONTROLLER = False         # Deprecated, True if using backup controller
WARP_AND_COLOR_PICKING = False  # Re-do warp & color selection
SELECTION_SCALE = 0.5           # Scale factor for selection windows (0.5 = half size)
IS_TRANSMITTING = False         # True if connected to live Huey
WEAPON_ON = False               # True if weapon motor should be on
SHOW_FRAME = True               # Show camera feed frames
IS_ORIGINAL_FPS = True          # Process every captured frame, False -> cap at FRAME_RATE
FRAME_RATE = 60                 # FPS used for algo stuff, update to expected FPS on your system.
DISPLAY_ANGLES = True           # Only show angles if SHOW_FRAME is True
SHOW_HUD = True                # Show heads-up display with FPS, speed, turn, frame number
SHOW_QUANTIZED_HUEY = True      # Display the quantized bounding box of Huey in separate window
COLOR_QUANTIZATION = True       # True if color quantization is on
CAN_RECOVER = True              # True if want recovery
CAMERA_STREAM = False           # True if using live camera stream, False if using pre-recorded video
SHEET_RUNTIME = True            # Save runtimes to a spreadsheet and generate a graph (install "Excel Viewer" VS Code extension)
SAVE_BBOXES = False             # Save bounding box images every BBOX_SAVE_FREQUENCY iterations
BBOX_SAVE_FREQUENCY = 10        # How often to save bounding box images (every n iterations)

# MODEL_NAME = "SmallComp"        # Used for comp, best accuracy if you have the compute for it.
MODEL_NAME = "NanoSizeVariant"  # Use with lower image size for faster performance, not much worse accuracy.

# Image size for object detection model, lower number -> faster, slightly worse accuracy.
OD_IMG_SIZE = 416               # 640 default, 416 fast, must be multiple of 32. Don't go below 320. 

# If model gives a bug, ask Aaron which model/image size to use for your system.
# TODO: Documentation for available models

folder = os.getcwd() + "/main_files"

# camera_number   = folder + "/test_videos/huey_vs_prince.mp4"
camera_number   = folder + "/test_videos/prince_v_huey_1080_30.mp4" # 30 FPS
# camera_number   = folder + "/test_videos/prince_v_huey_ultraHD_30.mp4" # Get from drive, 30 FPS
# camera_number = 1

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4
    
rs = RuntimeSheet(use=SHEET_RUNTIME)
# ------------------------------ BEFORE THE MATCH ------------------------------
def main():
    stream = None
    try:
        # 1. Start the capturing frame from the camera or pre-recorded video
        # 2. Capture initial frame by pressing '0'
        if CAMERA_STREAM:
            stream = CameraStream(camera_number).start()
            captured_image = key_frame(stream, CAMERA_STREAM, SELECTION_SCALE)
        else:
            cap = cv2.VideoCapture(camera_number)
            captured_image = key_frame(cap, CAMERA_STREAM, SELECTION_SCALE)

        # 3. Use the initial frame to get a new Homography Matrix and new colors
        if WARP_AND_COLOR_PICKING:
            warped_frame, homography_matrix = make_new_homography(captured_image, SELECTION_SCALE)
            selected_colors = make_new_colors(folder + "/selected_colors.txt", warped_frame)
        # 3. Or use the previously saved Homography Matrix and colors from the txt file
        else:
            warped_frame, homography_matrix = read_prev_homography(captured_image, folder + "/homography_matrix.txt")
            selected_colors = read_prev_colors(folder + "/selected_colors.txt")

        # Shift corner color hues closer to bot color to account for motion blur
        shift_factor = 0.15
        selected_colors[1][0] = int(selected_colors[1][0] + (selected_colors[0][0] - selected_colors[1][0]) * shift_factor)
        selected_colors[2][0] = int(selected_colors[2][0] + (selected_colors[0][0] - selected_colors[2][0]) * shift_factor)

        # Build warp maps from homography matrix for faster warping in the main loop
        map_x, map_y = get_warp_maps(homography_matrix)

        # 4. Initialize color quantization cv2
        if COLOR_QUANTIZATION:
            initialize_quantization()
        
        # 5. Defining all subsystem objects: ML, Corner, Algorithm, Transmission

        # Get predictor, if anything goes wrong here, call Aaron #TODO: Document better
        predictor = get_predictor(MODEL_NAME, OD_IMG_SIZE)

        corner_detection = RobotCornerDetection(selected_colors, False, False)
        algorithm = None
        # TODO: Figure out whether we need weapon_motor_group and JANK_CONTROLLER
        if IS_TRANSMITTING:
            ser, motor_group, weapon_motor_group = get_motor_groups(JANK_CONTROLLER, speed_motor_channel, turn_motor_channel, weapon_motor_channel)
            if WEAPON_ON:
                weapon_motor_group.move(1)
        
        cv2.destroyAllWindows()

        if WARP_AND_COLOR_PICKING:
            algorithm = first_run(predictor, warped_frame, SHOW_FRAME, corner_detection, selected_colors)
        else:
            algorithm = Ram()

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

        prev = ptime()
        last_frame = 0
        iteration = 0
        global_flipped = None
        start_time = ptime()
        is_paused = False  # Pause state for playback

        while (CAMERA_STREAM and stream.isOpened() and not stream.stopped) or (not CAMERA_STREAM and cap.isOpened()):
            time_elapsed = ptime() - prev
            
            # If paused, wait for key press to advance frame
            if is_paused:
                print("[PAUSED] Press any key to advance one frame, or 'p' to resume")
                while True:
                    key_press = cv2.waitKey(0)
                    if key_press == ord("p"):  # 'p' resumes playback
                        is_paused = False
                        print("Playback resumed")
                        break
                    else:  # Any other key advances one frame
                        break
            
            rs.start_iter()
            if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / FRAME_RATE) and (not CAMERA_STREAM or stream.frameCount() > last_frame):
                prev = ptime()

                iteration += 1

                # Save bboxes every BBOX_SAVE_FREQUENCY iterations if SAVE_BBOXES is True
                if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                    os.makedirs(f"{bb_output_dir}/frame_{iteration}", exist_ok=True)
                    frame_save_dir = os.path.join(bb_output_dir, f"frame_{iteration}")

                # Logs average of last 10 FPS
                if SHEET_RUNTIME:
                    if iteration > 11:
                        fps10 = 1.0 / ((prev - rs.get_row(-10)["Start Time"]) / 10.0)
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

                with rs.log_timing("PollKey"):
                    if SHOW_FRAME:
                        key = cv2.pollKey()
                        if key == ord("q"):  # Press Q on keyboard to exit
                            break
                        elif key == ord("f"): #F key to flip
                            print("Backup flipped key pressed")
                            if global_flipped is None:
                                global_flipped = True
                            else:
                                global_flipped = not global_flipped
                        elif key == ord("p"):  # Press P to toggle pause
                            is_paused = not is_paused
                            print(f"Playback {'paused' if is_paused else 'resumed'}")
                    else:
                        key = None


                with rs.log_timing("Warp"):
                    warped_frame = warp_map(frame, map_x, map_y)

                # 11. Run the Warped Image through Object Detection
                with rs.log_timing("Object Detection"):
                    detected_bots = predictor.predict(warped_frame)
                
                if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                    for bot in range(len(detected_bots["bots"])):
                        if detected_bots["bots"][bot]["img"] is not None:
                            cv2.imwrite(f"{frame_save_dir}/detected_bot_{bot}.png", detected_bots["bots"][bot]["img"])

                is_flipped = -1 if global_flipped else 1

                # 11.5 Quantize Colors
                with rs.log_timing("Color Quantization"):
                    if COLOR_QUANTIZATION:
                        detected_bots = quantize(detected_bots, selected_colors, show=False, is_flipped=is_flipped)

                if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                    for bot in range(len(detected_bots["bots"])):
                        if detected_bots["bots"][bot]["img"] is not None:
                            cv2.imwrite(f"{frame_save_dir}/quantized_bot_{bot}.png", detected_bots["bots"][bot]["img"])

                # 12. Run Object Detection's results through Corner Detection
                with rs.log_timing("Corner Detection"):
                    corner_detection.set_bots(detected_bots)
                    detected_bots_with_data = corner_detection.corner_detection_main()

                # Display quantized Huey if enabled (after corner detection identifies Huey)
                with rs.log_timing("Display Quantized Huey"):
                    if SHOW_QUANTIZED_HUEY and len(detected_bots["bots"]) > 0:
                        try:
                            if detected_bots_with_data and detected_bots_with_data.get("huey") and detected_bots_with_data["huey"].get("bbox") is not None:
                                huey_bbox = detected_bots_with_data["huey"]["bbox"]
                                # Find which bot index has this bbox and display only that one
                                for i, bot in enumerate(detected_bots["bots"]):
                                    if bot.get("bbox") is not None and np.array_equal(bot["bbox"], huey_bbox):
                                        if bot.get("img") is not None:
                                            cv2.imshow("Quantized Huey", bot["img"])
                                        
                                        # Display Hue-only version (Hue isolated, S=255, V=255)
                                        try:
                                            pt1, pt2 = huey_bbox
                                            x1, y1 = int(pt1[0]), int(pt1[1])
                                            x2, y2 = int(pt2[0]), int(pt2[1])
                                            
                                            # Clamp to frame bounds
                                            h_f, w_f = warped_frame.shape[:2]
                                            x1, x2 = max(0, x1), min(w_f, x2)
                                            y1, y2 = max(0, y1), min(h_f, y2)
                                            
                                            if x2 > x1 and y2 > y1:
                                                crop = warped_frame[y1:y2, x1:x2]
                                                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                                                h, s, v = cv2.split(hsv_crop)
                                                s.fill(255)
                                                v.fill(255)
                                                hue_only = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
                                                cv2.imshow("Hue Only Huey", hue_only)
                                        except Exception as e:
                                            pass
                                            
                                        break
                        except Exception as e:
                            pass

                with rs.log_timing("Algorithm"):
                    move_dictionary = algorithm.ram_ram(detected_bots_with_data, CAN_RECOVER, fps=FRAME_RATE, key=key)
                
                with rs.log_timing("Display"):
                    if DISPLAY_ANGLES:
                        # Moved from inside predict code to keep bb images clean of annotations.

                        warped_frame = predictor.show_predictions(warped_frame, detected_bots)

                        if SHOW_HUD:
                            warped_frame = draw_hud(warped_frame, fps10=fps10, move_dictionary=move_dictionary, iteration=iteration)
                        
                        final_image = display_angles(detected_bots_with_data, move_dictionary, warped_frame, is_recovering=algorithm.is_recovering, is_backing=algorithm.is_backing, against_wall=algorithm.against_wall, moving_forward=algorithm.moving_forward, is_flipped = is_flipped, centroids=corner_detection.centroids)

                        if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                            cv2.imwrite(f"{frame_save_dir}/final_image_{iteration}.png", final_image)

                # 14. Transmitting the motor values to Huey's if we're using a live video
                with rs.log_timing("Transmission"):
                    if IS_TRANSMITTING:
                        speed = move_dictionary["speed"]
                        turn = move_dictionary["turn"]
                        # print(f"Speed: {speed}")
                        # print(f"Turn: {turn}")
                        motor_group.move(speed*is_flipped, turn * -1)
                
                rs.dump()

            elif DISPLAY_ANGLES:
                display_angles(None, None, warped_frame)

            if SHOW_FRAME and not DISPLAY_ANGLES:
                display_frame = warped_frame
                if SHOW_HUD:
                    display_frame = draw_hud(display_frame, fps10=fps10, move_dictionary=move_dictionary, iteration=iteration)
                cv2.imshow("Bounding boxes (no angles)", display_frame)

        if CAMERA_STREAM:
            stream.stop()
        print("============================")
        print("Video finished successfully!")

        if SHOW_FRAME:
            cv2.destroyAllWindows()
            if SHOW_QUANTIZED_HUEY:
                try:
                    cv2.destroyWindow("Quantized Huey")
                    cv2.destroyWindow("Hue Only Huey")
                except:
                    pass

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")
    except Exception as exception:
        print("UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP:", exception)
    finally:

        ## Newbie squadron trial
        try:
            color_df = pd.DataFrame(corner_detection.color_percentage_rows)
            color_df.to_csv("color_output.csv", index=True)
            # color_percentages_graphing.makeGraph()
        except Exception as color_exception:
            print("Data collection failed:", color_exception)

        if IS_TRANSMITTING: # Motors need to be cleaned up correctly
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

        rs.save("runtimesheet")

if __name__ == "__main__":
    main()
