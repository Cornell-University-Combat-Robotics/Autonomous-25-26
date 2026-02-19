import os
import time

# from line_profiler import LineProfiler
import pandas as pd
import cv2
import torch

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

# ------------------------------ GLOBAL VARIABLES ------------------------------

MATT_LAPTOP = torch.cuda.is_available()             # True if running on Matt's laptop
JANK_CONTROLLER = False         # True if using backup controller
COMP_SETTINGS = False           # Competition mode (no visuals, optimized speed)
WARP_AND_COLOR_PICKING = True   # Re-do warp & color selection
IS_TRANSMITTING = False         # True if connected to live Huey
WEAPON_ON = False                # True if weapon motor should be on

# SET THIS HOE FIX IT 

SHOW_FRAME = True               # Show camera feed frames
IS_ORIGINAL_FPS = True         # Process every captured frame

DISPLAY_ANGLES = True     # Only show angles if frames a
COLOR_QUANTIZATION = True       # True if color quantization is on
CAN_RECOVER = True             # True if want recovery
# PROFILE_LINES = False            # True to display timing info for functions
CAMERA_STREAM = False
SHEET_RUNTIME = True
#TODO: don't recover on first frame

SRT = SHEET_RUNTIME

if COMP_SETTINGS:
    SHOW_FRAME = False
    DISPLAY_ANGLES = False
    MATT_LAPTOP = True   # Force TensorRT optimization on Matt's laptop

folder = os.getcwd() + "/main_files"
frame_rate = 60
# camera_number = folder + "/test_videos/trimmed_huey_redshift.mp4"
# camera_number = folder + "/test_videos/nhrl_arena.mp4"
# camera_number = folder + "/test_videos/huey_blushy.mp4"
# camera_number = folder + "/test_videos/huey_hell.mp4"
# camera_number = folder + "/test_videos/crude_rot_huey.mp4"
# camera_number = folder + "/test_videos/two_huey_real_cage_800.mp4"
camera_number   = folder + "/test_videos/huey_vs_prince.mp4"
# camera_number = 1

class RuntimeSheet:
    # Used for saving runtimes to a spreadsheet
    def __init__(self, use):
        self.init_time = time.perf_counter()
        self.sheet = []
        self.row = {"Start Time":time.perf_counter()}
        self.use = use

    def log(self, name, start_time):
        if self.use:
            self.row[name] = time.perf_counter()-start_time
    
    def start_iter(self):
        if self.use:
            self.row = {"Start Time":time.perf_counter()}

    def dump(self):
        if self.use:
            self.row["End Time"] = time.perf_counter()
            self.row["Elapsed Time"] = self.row["End Time"] - self.row["Start Time"]
            self.sheet.append(self.row)

    def save(self, output_name):
        if self.use:
            df = pd.DataFrame(self.sheet)
            df.to_csv(output_name)


if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

# if PROFILE_LINES:
#     profiler = LineProfiler()

#     def profile(func):
#         def inner(*args, **kwargs):
#             profiler.add_function(func)
#             profiler.enable_by_count()
#             return func(*args, **kwargs)
#         return inner
# else:
#     def profile(func):
#         return func
    
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
        predictor = get_predictor(MATT_LAPTOP)
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

        st = RuntimeSheet(SRT)

        # ----------------------------------------------------------------------
        # 8. Match begins
        if CAMERA_STREAM:
            if stream.isOpened() == False:
                print("Error opening video file" + "\n")
        else:
            if cap.isOpened() == False:
                print("Error opening video file" + "\n")
        prev = time.perf_counter()
        last_frame = 0
        fps_time = time.perf_counter()
        fps_frame = 0
        iteration = 0
        global_flipped = None

        while (CAMERA_STREAM and stream.isOpened() and not stream.stopped) or (not CAMERA_STREAM and cap.isOpened()):
            time_elapsed = time.perf_counter() - prev
            fps = 1/time_elapsed
            # print("FPS: " + str(fps))

            # 10. Warp image using the Homography Matrix
            rs.start_iter()
            if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate) and (not CAMERA_STREAM or stream.frameCount() > last_frame):
                # print("FPS: " + str(1/time_elapsed))
                prev = time.perf_counter()

                iteration = iteration + 1

                if time.perf_counter() - fps_time > 1.0:
                    print(f"Frames in last 1 second: {iteration - fps_frame}")
                    print(f"FPS over last 1 second: {(iteration - fps_frame)/1.0}")
                    fps_frame = iteration
                    fps_time = time.perf_counter()

                t = time.perf_counter()
                if CAMERA_STREAM:
                    ret, frame = stream.read()
                    # print("Frame number: " + str(stream.frameCount()))
                    last_frame = stream.frameCount()
                else: ret, frame = cap.read()
                rs.log("Frame Read", t)

                
                if not ret:
                    print("Failed to capture image" + "\n")
                    break

                t = time.perf_counter()
                
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

                rs.log("Waitkey 1", t)
                
                t = time.perf_counter()
                warped_frame = warp(frame, homography_matrix)
                rs.log("Warp", t)

                # 11. Run the Warped Image through Object Detection
                t = time.perf_counter()
                # detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
                detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)

                rs.log("Object Detection", t)

                if global_flipped == True:
                    is_flipped = -1
                else:
                    is_flipped = 1

                # 11.5 Quantize those mf colors
                t = time.perf_counter()
                if COLOR_QUANTIZATION:
                    # if iteration % 120 == 0:
                    #     detected_bots = quantize(detected_bots, selected_colors, show=True, is_flipped=is_flipped)
                    # else:
                    detected_bots = quantize(detected_bots, selected_colors, show=False, is_flipped=is_flipped)
                rs.log("Color Quant", t)

                #indonesia.set_bots(detected_bots)
                corner_detection.set_bots(detected_bots)
                # 12. Run Object Detection's results through Corner Detection
                t = time.perf_counter()
                detected_bots_with_data = corner_detection.corner_detection_main()
                rs.log("CD Main", t)

                t = time.perf_counter()
                move_dictionary = algorithm.ram_ram(detected_bots_with_data, CAN_RECOVER, fps=frame_rate, key=key)
                rs.log("Algorithm", t)

                if DISPLAY_ANGLES:
                    t = time.perf_counter()
                    display_angles(detected_bots_with_data, move_dictionary, warped_frame, is_recovering=algorithm.is_recovering, is_backing=algorithm.is_backing, against_wall=algorithm.against_wall, moving_forward=algorithm.moving_forward, is_flipped = is_flipped, centroids=corner_detection.centroids)
                    rs.log("Display Angles", t)

                # 14. Transmitting the motor values to Huey's if we're using a live video
                if IS_TRANSMITTING:
                    t = time.perf_counter()
                    speed = move_dictionary["speed"]
                    turn = move_dictionary["turn"]
                    # print(f"Speed: {speed}")
                    # print(f"Turn: {turn}")
                    if turn * -1 > 0:
                        motor_group.move(speed*is_flipped, turn * -1)
                    else:
                        motor_group.move(speed*is_flipped, turn * -1)
                    rs.log("Transmission", t)
                
                rs.dump()

            elif DISPLAY_ANGLES:
                # t = time.perf_counter()
                display_angles(None, None, warped_frame)
                # rs.log("Elif Display Angles", t)
                # rs.dump()
                # time.sleep(0.0005)
                # continue

            if SHOW_FRAME and not DISPLAY_ANGLES:
                # t = time.perf_counter()
                cv2.imshow("Bounding boxes (no angles)", warped_frame)
                # rs.log("Show No Angles", t)
                # rs.dump()

            

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

        # if PROFILE_LINES:
        #     profiler.print_stats(output_unit=1e-03)

        rs.save("runtimesheet.csv")

if __name__ == "__main__":
    main()
