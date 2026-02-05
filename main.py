import os
import time

from line_profiler import LineProfiler
import pandas as pd
import cv2

from camera_stream import CameraStream
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
from main_helpers import key_frame, read_prev_homography, make_new_homography, read_prev_colors, make_new_colors, get_predictor, get_motor_groups, first_run, display_angles, draw_yaw_text
from sensors.imu_class import IMU_sensor
from sensors.imu_class import IMUReadError

# ------------------------------ GLOBAL VARIABLES ------------------------------

MATT_LAPTOP = False             # True if running on Matt's laptop
JANK_CONTROLLER = False         # True if using backup controller
COMP_SETTINGS = False           # Competition mode (no visuals, optimized speed)
WARP_AND_COLOR_PICKING = True   # Re-do warp & color selection
IS_TRANSMITTING = True         # True if connected to live Huey
SHOW_FRAME = True               # Show camera feed frames
IS_ORIGINAL_FPS = True          # Process every captured frame
DISPLAY_ANGLES = SHOW_FRAME     # Only show angles if frames a
COLOR_QUANTIZATION = True        # True if color quantization is on
IMU_ENABLED = True              # True if IMU is connected
CAN_RECOVER = False             # True if want recovery
PROFILE_LINES = False           # True to display timing info for functions
CAMERA_STREAM = False
#TODO: don't recover on first frame

if COMP_SETTINGS:
    SHOW_FRAME = False
    DISPLAY_ANGLES = False
    MATT_LAPTOP = True   # Force TensorRT optimization on Matt's laptop

folder = os.getcwd() + "/main_files"
frame_rate = 30
# camera_number = folder + "/test_videos/kabedon_huey.mp4"
# camera_number = folder + "/test_videos/kabedon_huey.mp4"
# camera_number = folder + "/test_videos/lazy_huey.mp4"
# camera_number = folder + "/test_videos/green_huey_demo.mp4"
camera_number = 0

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

if PROFILE_LINES:
    profiler = LineProfiler()

    def profile(func):
        def inner(*args, **kwargs):
            profiler.add_function(func)
            profiler.enable_by_count()
            return func(*args, **kwargs)
        return inner
else:
    def profile(func):
        return func
# ------------------------------ BEFORE THE MATCH ------------------------------
@profile
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
        # 4. Or use the previously saved Homography Matrix and colors from the txt file
        else:
            warped_frame, homography_matrix = read_prev_homography(captured_image, folder + "/homography_matrix.txt")
            selected_colors = read_prev_colors(folder + "/selected_colors.txt")

        # 4. Initialize color quantization cv2
        if COLOR_QUANTIZATION:
            initialize_quantization()
        
        # 5. Defining all subsystem objects: ML, Corner, Algorithm, Transmission
        predictor = get_predictor(MATT_LAPTOP)
        corner_detection = RobotCornerDetection(selected_colors, False, False)
        algorithm = None
        if IMU_ENABLED:
            imu_sensor = IMU_sensor()
            cali_yaw = 0

        # TODO: Figure out whether we need weapon_motor_group and JANK_CONTROLLER
        if IS_TRANSMITTING:
            ser, motor_group, weapon_motor_group = get_motor_groups(JANK_CONTROLLER, speed_motor_channel, turn_motor_channel, weapon_motor_channel)
        
        cv2.destroyAllWindows()

        if WARP_AND_COLOR_PICKING:
            algorithm = first_run(predictor, warped_frame, SHOW_FRAME, corner_detection, selected_colors)
        else:
            algorithm = Ram()

        # ----------------------------------------------------------------------
        # 8. Match begins
        if CAMERA_STREAM:
            if stream.isOpened() == False:
                print("Error opening video file" + "\n")
        else:
            if cap.isOpened() == False:
                print("Error opening video file" + "\n")
        prev = 0
        last_frame = 0

        while (CAMERA_STREAM and stream.isOpened() and not stream.stopped) or (not CAMERA_STREAM and cap.isOpened()):
            time_elapsed = time.perf_counter() - prev
            fps = 1/time_elapsed
            print("FPS: " + str(fps))
            # 10. Warp image using the Homography Matrix
            if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate) and (not CAMERA_STREAM or stream.frameCount() > last_frame):
                if IMU_ENABLED:
                    try:
                        cali_yaw = imu_sensor.get_yaw_uncali()
                    except IMUReadError as ex:
                        # print(f"🟥 Error: {ex}") xd rawr
                        pass
                    except KeyError as ex:
                        # print(f"🟥 Error: {ex}")
                        pass
            
                print("FPS: " + str(1/time_elapsed))
                prev = time.perf_counter()
                if CAMERA_STREAM:
                    ret, frame = stream.read()
                    print("Frame number: " + str(stream.frameCount()))
                    last_frame = stream.frameCount()
                else: ret, frame = cap.read()

                if not ret:
                    print("Failed to capture image" + "\n")
                    break

                if SHOW_FRAME:
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q on keyboard to exit
                        break
                
                warped_frame = warp(frame, homography_matrix)

                # 11. Run the Warped Image through Object Detection
                detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)

                # 11.5 Quantize those mf colors
                if COLOR_QUANTIZATION:
                    detected_bots = quantize(detected_bots, selected_colors, show=False)

                #indonesia.set_bots(detected_bots)
                corner_detection.set_bots(detected_bots)
                # 12. Run Object Detection's results through Corner Detection
                detected_bots_with_data = corner_detection.corner_detection_main()
                print(detected_bots_with_data)
                print("📐corner works")
                is_flipped = 1
                if IMU_ENABLED:
                    try:
                        if detected_bots_with_data.get("huey") is not None:
                            if detected_bots_with_data.get("huey").get("orientation") is not None:
                                print(f"before cali yaw: {cali_yaw} and {detected_bots_with_data.get("huey").get("orientation")}")
                                imu_sensor.calibrate_yaw(detected_bots_with_data.get("huey").get("orientation"), cali_yaw)
                            else:
                                yaw = imu_sensor.get_yaw_continuous()
                                detected_bots_with_data["huey"]["orientation"] = yaw
                                print(f"yaw = {yaw}")
                        is_flipped = imu_sensor.get_upside_down_continuous()
                        print(f"flipped = {is_flipped}")
        
                        draw_yaw_text(warped_frame,yaw,is_flipped)
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
                # move_dictionary = algorithm.ram_ram(detected_bots_with_data)
                move_dictionary = algorithm.ram_ram(detected_bots_with_data, CAN_RECOVER, fps=fps)
                
                if DISPLAY_ANGLES:
                    display_angles(detected_bots_with_data, move_dictionary, warped_frame, is_recovering=algorithm.is_recovering, is_backing=algorithm.is_backing, against_wall=algorithm.against_wall, moving_forward=algorithm.moving_forward, centroids=corner_detection.centroids)

                # 14. Transmitting the motor values to Huey's if we're using a live video
                if IS_TRANSMITTING:
                    speed = move_dictionary["speed"]
                    turn = move_dictionary["turn"]
                    print(f"is flipped? {is_flipped}")
                    if turn * -1 > 0:
                        motor_group.move(speed * 0.8*is_flipped, turn * -1 * 0.55 + 0.2)
                    else:
                        motor_group.move(speed * 0.8*is_flipped, turn * -1 * 0.55 - 0.2)

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

        if PROFILE_LINES:
            profiler.print_stats(output_unit=1e-03)

if __name__ == "__main__":
    main()
