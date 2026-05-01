import os
import time
import threading
import json
from collections import deque

import pandas as pd
import cv2
import numpy as np
from time import perf_counter as ptime

from camera_stream import CameraStream
from runtimesheet.runtimesheet import RuntimeSheet
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
    draw_yaw_text
)
from warp_main import get_warp_maps
from warp_main import warp_map
from sensors.imu_class import IMU_sensor
from sensors.imu_class import IMUReadError

# ------------------------------ GLOBAL SETTINGS ------------------------------
# Keep one option active per setting. Commented lines directly below are common alternatives.

# Run mode (uncomment exactly one)
# MODE = "comp"
MODE = "live"
# MODE = "video"
# MODE = "custom"

# Core behavior
WARP_AND_COLOR_PICKING = True
DISPLAY_SCALE = 0.5  # 1.0 for full-size display, 0.5 for easier 1080p selection
CAN_RECOVER = True
CAN_RECOVER = True
BLACKOUT = True
COLOR_QUANTIZATION = True  # Should almost always stay True
CAMERA_STREAM = True     # Frame capture thread (must be False for videos)
IMU_ENABLED = True    # Set to True to enable IMU integration (if hardware is available)
USE_TRACKING = True       # Use tracking-based predictor instead of running detection on every frame (requires more resources)
DETECTION_CONFIDENCE = 0.25  # Ultralytics default is 0.25; Try lower values

# Logging / debug outputs
SHEET_RUNTIME = True
rs = RuntimeSheet(use=SHEET_RUNTIME)

# Display toggles
SHOW_FRAME = True
DISPLAY_ANGLES = True  # Only applies when SHOW_FRAME is True
SHOW_HUD = True
SHOW_QUANTIZED_HUEY = True

# Hardware / controls
JANK_CONTROLLER = False  # Deprecated backup controller path
IS_TRANSMITTING = True
WEAPON_ON = False

# Frame timing
IS_ORIGINAL_FPS = True
FRAME_RATE = 120
# FRAME_RATE = 60  # Common for video testing

# Model selection
# MODEL_NAME = "SmallComp"       # Best accuracy if compute allows
# MODEL_NAME = "NanoSizeVariant" # Faster, slightly lower accuracy
MODEL_NAME = "Nano320Temp"       # Trained with match images at 320 size
OD_IMG_SIZE = 320                # Must be multiple of 32, avoid below 320

if MODE == "comp" or MODE == "live":
    IS_TRANSMITTING = True         # True to send transmissions to live Huey via Arduino    
    IS_ORIGINAL_FPS = True         # Process every captured frame, False -> cap at FRAME_RATE, only TRUE for Live
    FRAME_RATE = 120               # Used in recovery/algo  
    CAMERA_STREAM = True           # True to run frame capture in a seperate thread, always false for videos

    if MODE == "comp":
        WEAPON_ON = True   

    else: # MODE == "live"
        WEAPON_ON = False

elif MODE == "video":
    IS_TRANSMITTING = False         # True to send transmissions to live Huey via Arduino
    WEAPON_ON = False
    IS_ORIGINAL_FPS = False         # Process every captured frame, False -> cap at FRAME_RATE, only TRUE for Live
    FRAME_RATE = 30                 # Manually set frame rate for videos
    CAMERA_STREAM = False           # True to run frame capture in a seperate thread, always false for videos

elif MODE == "custom":
    # Custom mode allows you to specify all settings manually
    pass

else:
    raise ValueError(f"Invalid MODE '{MODE}'. Expected 'comp', 'live', 'video', or 'custom'.")

# ------------------------------ CAMERA / VIDEO INPUT ------------------------------

folder = os.getcwd() + "/main_files"
# Video options (uncomment one for MODE = "video")
# camera_number = folder + "/test_videos/crude_rot_huey.mp4"
# camera_number = folder + "/test_videos/huey_vs_prince.mp4"
# camera_number = folder + "/test_videos/huey_hell.mp4"
# camera_number = folder + "/test_videos/huey_in_n_out.mp4"
# camera_number = folder + "/test_videos/cicero_corners_bzone.mov"
# camera_number = folder + "/test_videos/orbital_huey.mp4"
# camera_number = folder + "/test_videos/diagona_huey.mp4"
# camera_number = folder + "/test_videos/huey_backs.mp4"

# Webcam index (used for MODE = "live" or MODE = "comp")
camera_number = 0
# camera_number = 1

# Set to webcam if capturing frames in main loop.
# camera_type = "Video"
camera_type = "Webcam"

# ------------------------------ QUANTIZATION SETTINGS ------------------------------

quant_settings_file = "quant_settings.json"
with open(quant_settings_file, "r") as f:
    all_settings = json.load(f)

# Quantization Settings
quantization_settings = None
# quantization_settings = all_settings["Green Huey"]
# quantization_settings = all_settings["Green Huey Area"]
# quantization_settings = all_settings["Purple Huey"]

DEFAULT_AREA_THRESHOLD = 15
area_threshold = DEFAULT_AREA_THRESHOLD
if quantization_settings and "area_theshold" in quantization_settings:
    area_threshold = quantization_settings["area_theshold"]

# ------------------------------ BEFORE THE MATCH ------------------------------

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

# Threading globals
frame_buffer = deque(maxlen=1)
stop_event = threading.Event()
shared_state_lock = threading.Lock()
# Shared state for controls passed from UI thread to Perception thread
shared_state = {"key": None, "flipped": None,
                "paused": False, "skip_frame": False, "weapon_on": WEAPON_ON}

prev_sensor_val = 0
curr_sensor_val = 0
total_sensor_val = 0
cali_yaw = 0

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

            if camera_type == "Webcam":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        initialize_quantization()

        # Get predictor, if anything goes wrong here, call Aaron #TODO: Document better
        predictor = get_predictor(MODEL_NAME, OD_IMG_SIZE)

        if IMU_ENABLED:
            imu_sensor = IMU_sensor()
            # cali_yaw = 0
        

        # Initialize corner detection
        corner_detection = RobotCornerDetection(selected_colors, False, False, BLACKOUT=BLACKOUT, thresh=0.4, frame_rate = FRAME_RATE)

        # Initialize transmission TODO: Figure out whether we need weapon_motor_group and JANK_CONTROLLER
        if IS_TRANSMITTING:
            ser, motor_group, weapon_motor_group = get_motor_groups(
                False, speed_motor_channel, turn_motor_channel, weapon_motor_channel)
            # if WEAPON_ON:
            #     weapon_motor_group.move(1)

        cv2.destroyAllWindows()

        # # Initialize algorithm
        # if WARP_AND_COLOR_PICKING:
        #     algorithm = first_run(predictor, warped_frame,
        #                           SHOW_FRAME, corner_detection, selected_colors)
        # else:
        #     algorithm = Ram()

        algorithm = first_run(predictor, warped_frame, SHOW_FRAME, corner_detection, selected_colors, area_threshold)

        ### TODO: call dynamic threshold here

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
        # This is all of our processing code minus the display of the images.
        # Any image displays should modify the frame that is returned at the end of the loop.
        def perception_pipeline():
            global prev_sensor_val, curr_sensor_val, total_sensor_val, cali_yaw

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

                # Handle pause state with synchronized reads/writes.
                with shared_state_lock:
                    is_paused = shared_state["paused"]
                    should_skip_one = shared_state["skip_frame"]
                    if is_paused and should_skip_one:
                        shared_state["skip_frame"] = False
                if is_paused and not should_skip_one:
                    time.sleep(0.01)
                    continue

                time_elapsed = ptime() - prev

                rs.start_iter()
                # TODO: Fix frame rate logic
                if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / FRAME_RATE) and (not CAMERA_STREAM or stream.frameCount() > last_frame):
                    prev = ptime()
                    iteration += 1

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
                    with shared_state_lock:
                        key = shared_state["key"]
                        manual_is_flipped = -1 if shared_state["flipped"] else 1
                        weapon_on_this_frame = shared_state["weapon_on"]
                        # Clear transient key so one press is consumed once.
                        shared_state["key"] = None

                    if IMU_ENABLED:
                        is_flipped = manual_is_flipped
                        try:
                            is_flipped = imu_sensor.get_upside_down_continuous()
                            with shared_state_lock:
                                shared_state["flipped"] = is_flipped == -1
                        except IMUReadError as ex:
                            print(f"🟥 Error reading IMU flip state, using manual flip: {ex}")
                        except KeyError as ex:
                            print(f"🟥 Error reading IMU flip state, using manual flip: {ex}")
                        except Exception as ex:
                            print(f"🟥 Error reading IMU flip state, using manual flip: {ex}")
                    else:
                        is_flipped = manual_is_flipped

                    # Warp image to homography matrix using maps
                    with rs.log_timing("Warp"):
                        warped_frame = warp_map(frame, map_x, map_y)

                    if IMU_ENABLED:
                        try:
                            cali_yaw = imu_sensor.get_yaw_uncali()
                            curr_sensor_val = cali_yaw
                            print(f"Total sensor value: {total_sensor_val}")
                            if prev_sensor_val == curr_sensor_val:
                                total_sensor_val += 1
                            else:
                                total_sensor_val = 0

                            prev_sensor_val = curr_sensor_val

                        except IMUReadError as ex:
                            # print(f"🟥 Error: {ex}") xd rawr
                            pass
                        except KeyError as ex:
                            # print(f"🟥 Error: {ex}")
                            pass
                        except KeyboardInterrupt as e:
                            raise(e)
                        except Exception as e:
                            print(f"error from imu: {e}")
                            pass

                    # 11. Run the Warped Image through Object Detection
                    # Internal timings (Preprocess, Inference, etc.) are handled inside predict()
                    with rs.log_timing("Object Detection"):
                        detected_bots = predictor.predict(
                            warped_frame, confidence_threshold=DETECTION_CONFIDENCE, track=USE_TRACKING)

                    # 11.5 Quantize Colors
                    with rs.log_timing("Color Quantization"):
                        detected_bots = quantize(
                            detected_bots, selected_colors, show=False, is_flipped=is_flipped, settings=quantization_settings)

                    # 12. Run Object Detection's results through Corner Detection
                    with rs.log_timing("Corner Detection"):
                        corner_detection.set_bots(detected_bots)
                        print("called corner main")
                        detected_bots_with_data, confidence = corner_detection.corner_detection_main(area_threshold, algorithm.huey_previous_orientations, is_flipped=is_flipped, tolerance=15)
                        print("Confidence 😤😤😤: ", confidence)

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

                    if IMU_ENABLED:
                        try:
                            # print("detected bots with data: ", detected_bots_with_data)
                            
                            # if detected_bots_with_data.get("huey") is not None and detected_bots_with_data.get("huey") != {}:
                            # print(q)
                            if total_sensor_val <= 400: 
                                if detected_bots_with_data and detected_bots_with_data.get("huey"):
                                    print(f"DETECTED BOTS WITH DATA {detected_bots_with_data.get('huey')}")
                                    if (detected_bots_with_data.get("huey").get("orientation") is not None) and confidence:
                                        #print(f"before cali yaw: {cali_yaw} and {detected_bots_with_data.get("huey").get("orientation")}")
                                        imu_sensor.calibrate_yaw(detected_bots_with_data.get("huey").get("orientation"), cali_yaw)
                                        print("CALLIBRATING")
                                        yaw = 0
                                    if (detected_bots_with_data.get("huey")) and (corner_detection.corner_method < 2 or corner_detection.is_diagonal):
                                        print(f"USING SENSORS USING SENSORS USING SENSORS")
                                        yaw = imu_sensor.get_yaw_continuous()
                                        detected_bots_with_data["huey"]["orientation"] = yaw
                                        print(f"yaw = {yaw}")
                                        draw_yaw_text(warped_frame,yaw,is_flipped)
                            # is_flipped = imu_sensor.get_upside_down_continuous()
                            print(f"flipped = {is_flipped}")
                            # print("detected bots with data: ", detected_bots_with_data)
            
                            # draw_yaw_text(warped_frame,yaw,is_flipped)
                        except IMUReadError as ex:
                            print(f"🟥 Error: {ex}")
                            print(" 🟢 using cd orientation 🟢 ")
                            pass
                        except KeyError as ex:
                            print(f"🟥 Error: {ex}")
                            pass
                        except Exception as ex:
                            print("🦅 WTF is Happening 🦅")
                            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                            message = template.format(type(ex).__name__, ex.args)
                            print(message)
                            raise(ex)        

                    with rs.log_timing("Algorithm"):
                        move_dictionary = algorithm.ram_ram(
                            detected_bots_with_data, CAN_RECOVER, fps=FRAME_RATE, key=key)
                        

                    # 14. Transmitting the motor values to Huey's if we're using a live video
                    with rs.log_timing("Transmission"):
                        if IS_TRANSMITTING:
                            speed = move_dictionary["speed"]
                            turn = move_dictionary["turn"]
                            motor_group.move(speed*is_flipped, turn * -1)
                            if WEAPON_ON:
                                weapon_motor_group.move(
                                    0.3 if weapon_on_this_frame else 0) # 0.8 before

                    # Prepare Main Display Image
                    main_display_img = None
                    with rs.log_timing("Display"):
                        if DISPLAY_ANGLES:
                            warped_frame = predictor.show_predictions(
                                warped_frame, detected_bots)
                            if SHOW_HUD:
                                # Uncomment this to get all the stats
                                warped_frame = draw_hud(
                                    warped_frame, fps10=fps10, move_dictionary=move_dictionary, iteration=iteration)
                                # Uncomment this to get only frame rate:
                                # warped_frame = draw_hud(
                                #     warped_frame, iteration=iteration)

                            # Call display_angles with show=False to get the image without displaying
                            main_display_img = display_angles(detected_bots_with_data, move_dictionary, warped_frame, is_recovering=algorithm.is_recovering, is_backing=algorithm.is_backing,
                                                              against_wall=algorithm.against_wall, moving_forward=algorithm.moving_forward, is_flipped=is_flipped, weapon_on=weapon_on_this_frame, centroids=corner_detection.centroids, is_confident=confidence, show=False)

                        elif SHOW_FRAME:
                            display_frame = warped_frame
                            if SHOW_HUD:
                                display_frame = draw_hud(
                                    display_frame, fps10=fps10, move_dictionary=move_dictionary, iteration=iteration)
                            main_display_img = display_frame

                    # Update Frame Buffer
                    frame_buffer.append({ "main": main_display_img, "huey": huey_display_img})
                    rs.dump()
                else:
                    # Prevent hot-spin while waiting for next frame/time budget.
                    # Without this, the loop burns CPU doing no useful work.
                    time.sleep(0.001)
            

        # Start the Perception Thread
        perception_thread = threading.Thread(target=perception_pipeline, daemon=True)
        perception_thread.start()

        # ----------------------------------------------------------------------
        # Display UI Loop (Runs in Main Thread)
        while not stop_event.is_set():
            if frame_buffer:
                frames = frame_buffer[0]

                if frames["main"] is not None and SHOW_FRAME:
                    name = "Battle with Predictions" if DISPLAY_ANGLES else "Bounding boxes (no angles)"
                    cv2.imshow(name, frames["main"])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                if frames["huey"] is not None and SHOW_QUANTIZED_HUEY:
                    cv2.imshow("Quantized Huey", frames["huey"])

            # waitKeyEx(1) pumps GUI events reliably and captures key presses.
            key = cv2.pollKey()

            if key != -1:
                key_8bit = key & 0xFF
                if key_8bit == ord("q"):
                    stop_event.set()
                elif key_8bit == ord("f"):
                    print("Backup flipped key pressed")
                    with shared_state_lock:
                        if shared_state["flipped"] is None:
                            shared_state["flipped"] = True
                        else:
                            shared_state["flipped"] = not shared_state["flipped"]
                        if shared_state["paused"]:
                            shared_state["skip_frame"] = True
                elif key_8bit == ord("p"):
                    with shared_state_lock:
                        shared_state["paused"] = not shared_state["paused"]
                        shared_state["skip_frame"] = False
                        paused_now = shared_state["paused"]
                    print(
                        f"Playback {'paused' if paused_now else 'resumed'}")
                elif key_8bit == ord("w"):
                    with shared_state_lock:
                        shared_state["weapon_on"] = not shared_state["weapon_on"]
                        weapon_now = shared_state["weapon_on"]
                    print(
                        f"Weapon {'ON' if weapon_now else 'OFF'}")
                else:
                    with shared_state_lock:
                        if shared_state["paused"]:
                            # Any other key while paused skips one frame.
                            shared_state["skip_frame"] = True

                # Pass key to perception thread for algorithm hooks.
                with shared_state_lock:
                    shared_state["key"] = key_8bit

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
