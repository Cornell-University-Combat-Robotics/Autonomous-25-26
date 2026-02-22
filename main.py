import os
import time

import pandas as pd
import cv2
import torch
from time import perf_counter as ptime

from camera_stream import CameraStream
from runtimesheet import RuntimeSheet
import matplotlib.pyplot as plt
from algorithm.ram import Ram
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
from warp_main import warp

# ------------------------------ GLOBAL VARIABLES ------------------------------

# MATT_LAPTOP = False           # Deprecated, matt laptop handled by torch device checks
JANK_CONTROLLER = False         # Deprecated, True if using backup controller
WARP_AND_COLOR_PICKING = False  # Re-do warp & color selection
IS_TRANSMITTING = False         # True if connected to live Huey
WEAPON_ON = False               # True if weapon motor should be on
SHOW_FRAME = True               # Show camera feed frames
IS_ORIGINAL_FPS = True          # Process every captured frame
DISPLAY_ANGLES = True           # Only show angles if SHOW_FRAME is True
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
frame_rate = 30
# camera_number = folder + "/test_videos/trimmed_huey_redshift.mp4"
# camera_number = folder + "/test_videos/nhrl_arena.mp4"
# camera_number = folder + "/test_videos/huey_blushy.mp4"
# camera_number = folder + "/test_videos/huey_hell.mp4"
# camera_number = folder + "/test_videos/crude_rot_huey.mp4"
# camera_number = folder + "/test_videos/two_huey_real_cage_800.mp4"
camera_number   = folder + "/test_videos/huey_vs_prince.mp4"
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
            captured_image = key_frame(stream, CAMERA_STREAM)
        else:
            cap = cv2.VideoCapture(camera_number)
            captured_image = key_frame(cap, CAMERA_STREAM)

        # 3. Use the initial frame to get a new Homography Matrix and new colors
        if WARP_AND_COLOR_PICKING:
            warped_frame, homography_matrix = make_new_homography(captured_image)
            selected_colors = make_new_colors(folder + "/selected_colors.txt", warped_frame)
        # 3. Or use the previously saved Homography Matrix and colors from the txt file
        else:
            warped_frame, homography_matrix = read_prev_homography(captured_image, folder + "/homography_matrix.txt")
            selected_colors = read_prev_colors(folder + "/selected_colors.txt")

        # 4. Initialize color quantization cv2
        if COLOR_QUANTIZATION:
            initialize_quantization()
        
        # 5. Defining all subsystem objects: ML, Corner, Algorithm, Transmission

        # Get predictor, if anything goes wrong here, call Aaron 717-984-3250 #TODO: Document better
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

        st = RuntimeSheet(SHEET_RUNTIME)

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
        fps_time = ptime()
        fps_frame = 0
        iteration = 0
        global_flipped = None
        start_time = ptime()

        while (CAMERA_STREAM and stream.isOpened() and not stream.stopped) or (not CAMERA_STREAM and cap.isOpened()):
            time_elapsed = ptime() - prev
            fps = 1/time_elapsed

            # 10. Warp image using the Homography Matrix
            rs.start_iter()
            if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate) and (not CAMERA_STREAM or stream.frameCount() > last_frame):
                prev = ptime()

                iteration = iteration + 1

                # Save bboxes every BBOX_SAVE_FREQUENCY iterations if SAVE_BBOXES is True
                if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                    os.makedirs(f"{bb_output_dir}/frame_{iteration}", exist_ok=True)
                    frame_save_dir = os.path.join(bb_output_dir, f"frame_{iteration}")

                # Prints true FPS every second
                if ptime() - fps_time > 1.0:
                    print(f"Frames in last 1 second: {iteration - fps_frame}")
                    fps_frame = iteration
                    fps_time = ptime()

                # Logs average of last 10 FPS
                if iteration > 11:
                    fps10 = 1.0 / ((prev - rs.get_row(-10)["Start Time"]) / 10.0)
                else:
                    fps10 = 1.0 / ((prev - start_time) / iteration)
                rs.log("FPS10", fps10)

                # Grabs frame from camera thread if using camera stream, otherwise reads from video
                t = ptime()
                if CAMERA_STREAM:
                    ret, frame = stream.read()
                    # print("Frame number: " + str(stream.frameCount()))
                    last_frame = stream.frameCount()
                else: ret, frame = cap.read()
                rs.log("Frame Read", ptime() - t)

                if not ret:
                    print("Failed to capture image" + "\n")
                    break

                t = ptime() 
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
                else:
                    key = None
                rs.log("Pollkey", ptime() - t)
                
                t = ptime()
                warped_frame = warp(frame, homography_matrix)
                rs.log("Warp", ptime() - t)

                # 11. Run the Warped Image through Object Detection
                t = ptime()
                # detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
                detected_bots = predictor.predict(warped_frame)
                
                if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                    for bot in range(len(detected_bots["bots"])):
                        if detected_bots["bots"][bot]["img"] is not None:
                            cv2.imwrite(f"{frame_save_dir}/detected_bot_{bot}.png", detected_bots["bots"][bot]["img"])

                rs.log("Object Detection", ptime() - t)

                if global_flipped == True:
                    is_flipped = -1
                else:
                    is_flipped = 1

                # 11.5 Quantize Colors
                t = ptime()
                if COLOR_QUANTIZATION:
                    detected_bots = quantize(detected_bots, selected_colors, show=False, is_flipped=is_flipped)
                rs.log("Color Quant", ptime() - t)

                if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                    for bot in range(len(detected_bots["bots"])):
                        if detected_bots["bots"][bot]["img"] is not None:
                            cv2.imwrite(f"{frame_save_dir}/quantized_bot_{bot}.png", detected_bots["bots"][bot]["img"])

                corner_detection.set_bots(detected_bots)

                # 12. Run Object Detection's results through Corner Detection
                t = ptime()
                detected_bots_with_data = corner_detection.corner_detection_main()
                rs.log("CD Main", ptime() - t)

                t = ptime()
                move_dictionary = algorithm.ram_ram(detected_bots_with_data, CAN_RECOVER, fps=frame_rate, key=key)
                enemy_orientation = algorithm.enemy_orientation
                enemy_future_position = algorithm.enemy_future_position
                print (f"🛸ENEMY FUTPOS:🛸 {enemy_future_position}")
                rs.log("Algorithm", t)

                if DISPLAY_ANGLES:
                    # Moved from inside predict code to keep bb images clean of annotations.
                    warped_frame = predictor.show_predictions(warped_frame, detected_bots)

                    t = ptime()
                    final_image = display_angles(detected_bots_with_data, move_dictionary, warped_frame, enemy_orientation, enemy_future_position, is_recovering=algorithm.is_recovering, is_backing=algorithm.is_backing, against_wall=algorithm.against_wall, moving_forward=algorithm.moving_forward, is_flipped = is_flipped, centroids=corner_detection.centroids)
                    rs.log("Display Angles", ptime() - t)

                    if SAVE_BBOXES and iteration % BBOX_SAVE_FREQUENCY == 1:
                        cv2.imwrite(f"{frame_save_dir}/final_image_{iteration}.png", final_image)

                # 14. Transmitting the motor values to Huey's if we're using a live video
                if IS_TRANSMITTING:
                    t = ptime()
                    speed = move_dictionary["speed"]
                    turn = move_dictionary["turn"]
                    # print(f"Speed: {speed}")
                    # print(f"Turn: {turn}")
                    if turn * -1 > 0:
                        motor_group.move(speed*is_flipped, turn * -1)
                    else:
                        motor_group.move(speed*is_flipped, turn * -1)
                    rs.log("Transmission", ptime() - t)
                
                rs.dump()

            elif DISPLAY_ANGLES:
                display_angles(None, None, warped_frame)

            if SHOW_FRAME and not DISPLAY_ANGLES:
                cv2.imshow("Bounding boxes (no angles)", warped_frame)

        if CAMERA_STREAM:
            stream.stop()
        print("============================")
        print("Video finished successfully!")

        if SHOW_FRAME:
            cv2.destroyAllWindows()

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
