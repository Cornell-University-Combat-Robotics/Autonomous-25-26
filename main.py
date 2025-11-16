import os
import time
import cv2
from algorithm.ram import Ram
from corner_detection.corner_detection import RobotCornerDetection
from warp_main import warp
from main_helpers import key_frame, read_prev_homography, make_new_homography, read_prev_colors, make_new_colors, get_predictor, get_motor_groups, first_run, display_angles

# ------------------------------ GLOBAL VARIABLES ------------------------------

MATT_LAPTOP = False             # True if running on Matt's laptop
JANK_CONTROLLER = False         # True if using backup controller
COMP_SETTINGS = False           # Competition mode (no visuals, optimized speed)
WARP_AND_COLOR_PICKING = True   # Re-do warp & color selection
IS_TRANSMITTING = False         # True if connected to live Huey
SHOW_FRAME = True               # Show camera feed frames
IS_ORIGINAL_FPS = False         # Process every captured frame
DISPLAY_ANGLES = SHOW_FRAME     # Only show angles if frames are displayed

if COMP_SETTINGS:
    SHOW_FRAME = False
    DISPLAY_ANGLES = False
    MATT_LAPTOP = True   # Force TensorRT optimization on Matt's laptop

folder = os.getcwd() + "/main_files"
frame_rate = 50
# camera_number = folder + "/test_videos/kabedon_huey.mp4"
# camera_number = folder + "/test_videos/huey_demo3.mp4"
# camera_number = folder + "/test_videos/huey_duet_demo.mp4"
camera_number = folder + "/test_videos/tooth_divorcee_huey.mp4"
# camera_number = 1

if IS_TRANSMITTING:
    speed_motor_channel = 1
    turn_motor_channel = 3
    weapon_motor_channel = 4

# ------------------------------ BEFORE THE MATCH ------------------------------

def main(): # TODO: Add timing back (kernprof)
    try:
        # 1. Start the capturing frame from the camera or pre-recorded video
        # 2. Capture initial frame by pressing '0'
        print("0")
        cap = cv2.VideoCapture(camera_number)
        print("1")
        captured_image = key_frame(cap)
        print("2")

        # 3. Use the initial frame to get a new Homography Matrix and new colors
        if WARP_AND_COLOR_PICKING:
            warped_frame, homography_matrix = make_new_homography(captured_image)
            selected_colors = make_new_colors(folder + "/selected_colors.txt", warped_frame)
        # 3. Or use the previously saved Homography Matrix and colors from the txt file
        else:
            warped_frame, homography_matrix = read_prev_homography(captured_image, folder + "/homography_matrix.txt")
            selected_colors = read_prev_colors(folder + "/selected_colors.txt")

        # 5. Defining all subsystem objects: ML, Corner, Algorithm, Transmission
        predictor = get_predictor(MATT_LAPTOP)
        corner_detection = RobotCornerDetection(selected_colors, False, False)
        algorithm = None
        # TODO: Figure out whether we need weapon_motor_group and JANK_CONTROLLER
        if IS_TRANSMITTING:
            ser, motor_group, weapon_motor_group = get_motor_groups(JANK_CONTROLLER, speed_motor_channel, turn_motor_channel, weapon_motor_channel)
        
        cv2.destroyAllWindows()

        if WARP_AND_COLOR_PICKING:
            print("FIRST RUN")
            algorithm = first_run(predictor, warped_frame, SHOW_FRAME, corner_detection)
        else:
            print("NOT FIRST RUN")
            algorithm = Ram()

        print("First run is done")

        # ----------------------------------------------------------------------
        # 8. Match begins
        if cap.isOpened() == False:
            print("Error opening video file" + "\n")
        prev = 0

        while cap.isOpened():
            time_elapsed = time.perf_counter() - prev
            # 10. Warp image using the Homography Matrix
            if IS_ORIGINAL_FPS or time_elapsed > 1.0 / frame_rate:
                ret, frame = cap.read()

                if not ret:
                    print("Failed to capture image" + "\n")
                    break

                if SHOW_FRAME:
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q on keyboard to exit
                        print("exit" + "\n")
                        break
                
                prev = time.perf_counter()
                warped_frame = warp(frame, homography_matrix, 700, 700)

                # 11. Run the Warped Image through Object Detection
                detected_bots = predictor.predict(warped_frame, show=SHOW_FRAME, track=True)
                
                #indonesia.set_bots(detected_bots)
                corner_detection.set_bots(detected_bots)
                # 12. Run Object Detection's results through Corner Detection
                detected_bots_with_data = corner_detection.corner_detection_main()
                move_dictionary = algorithm.ram_ram(detected_bots_with_data)

                enemy_orientation = algorithm.enemy_orientation
                is_recovering = algorithm.is_recovering
                
                if DISPLAY_ANGLES:
                    display_angles(detected_bots_with_data, move_dictionary, warped_frame, enemy_orientation, is_recovering)

                # 14. Transmitting the motor values to Huey's if we're using a live video
                if IS_TRANSMITTING:
                    speed = move_dictionary["speed"]
                    turn = move_dictionary["turn"]
                    if turn * -1 > 0:
                        motor_group.move(speed * 0.8, turn * -1 * 0.55 + 0.2)
                    else:
                        motor_group.move(speed * 0.8, turn * -1 * 0.55 - 0.2)

            elif DISPLAY_ANGLES:
                display_angles(None, None, warped_frame)

            if SHOW_FRAME and not DISPLAY_ANGLES:
                cv2.imshow("Bounding boxes (no angles)", warped_frame)

        cap.release()
        print("============================")
        print("Video finished successfully!")

        if SHOW_FRAME:
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT CLEAN UP")
    except Exception as exception:
        print("UNKNOWN EXCEPTION FAILURE. PROCEEDING TO CLEAN UP:", exception)
    finally:
        if IS_TRANSMITTING: # Motors need to be cleaned up correctly
            try:
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
    main()
