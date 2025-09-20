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
from arrow.color_picker import ColorPicker
from arrow.arrow import Arrow
from machine.predict import YoloModel
from transmission.motors import Motor
from transmission.serial_conn import OurSerial
from warp_main import get_homography_mat, warp
from vid_and_img_processing.unfisheye import unfish
from vid_and_img_processing.unfisheye import prepare_undistortion_maps

"""
Gets first frame of the video and returns it. If frame can't be read or video isn't being 
processed will print the problem, and return captured_image as none. 
"""
def KeyFrame(cap, resize_factor):
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
                resized_image = cv2.resize(
                    captured_image, (0, 0), fx=resize_factor, fy=resize_factor)
                h, w = resized_image.shape[:2]
                map1, map2 = prepare_undistortion_maps(w, h)
                return captured_image
        else:
            print("Failed to read frame" + "\n")
            return captured_image
    cv2.destroyAllWindows()
    return captured_image

